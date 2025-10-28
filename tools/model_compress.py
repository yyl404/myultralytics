"""
压缩YOLO模型

用法:
    python tools/model_compress.py --model <model_path> --save_path <save_path> --pca_cache_path <pca_cache_path> --ratio <ratio> --layers <layers>

参数:
    --model <model_path>: 原始模型路径
    --save_path <save_path>: 保存压缩后的模型的路径
    --pca_cache_path <pca_cache_path>: 原始模型的PCA缓存路径
    --ratio <ratio>: 选择保留的PCA成分的累积方差比例
    --layers <layers>: 选择进行压缩的层id列表，如果为空，则压缩所有层
"""
import os
import joblib
import argparse
from copy import deepcopy
from tqdm import tqdm
import cv2
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, resize

from ultralytics import YOLO
from ultralytics.nn.modules import DoubleConv, Conv
from ultralytics.utils import LOGGER, YAML
from ultralytics.nn.tasks import yaml_model_load


def compress_conv_module(module, compressed_module, variances, means, components, ratio):
    """
    压缩卷积模块，返回近似等效的HourglassConv模块

    Args:
        module(Conv): 需要压缩的卷积模块
        variances(torch.Tensor): 原始模型的PCA方差
        means(torch.Tensor): 原始模型的PCA均值
        components(torch.Tensor): 原始模型的PCA成分
        ratio(float): 选择保留的PCA成分的累积方差比例

    Returns:
        None
    """
    assert isinstance(module, Conv), "Module must be a convolutional module"
    # 展开权重和偏置
    weight = module.conv.weight.reshape(module.conv.weight.shape[0], -1) # [c_out, c_in*k*k]
    if module.conv.bias is not None:
        bias = module.conv.bias.reshape(module.conv.bias.shape[0], -1) # [c_out, 1]
        weight = torch.concat([weight, bias], dim=1) # [c_out, c_in*k*k+1]
    # 计算对应于方差累积比例的成分数量
    explained_variance_ratio = torch.cumsum(variances, dim=0) / torch.sum(variances) # [n_components]
    n_components_compressed = torch.sum(explained_variance_ratio <= ratio).item() # int
    # 计算近似等效的级联卷积模块权重
    encoding_matrix = F.normalize(components[:n_components_compressed], p=2, dim=1).detach() # [n_components_compressed, c_in*k*k] | [n_components_compressed, c_in*k*k+1]
    decoding_matrix = (weight @ encoding_matrix.T).detach() # [c_out, n_components_compressed]
    # 获取卷积模块的参数
    c_in = module.conv.in_channels
    c_out = module.conv.out_channels
    p = module.conv.padding
    k = module.conv.kernel_size
    s = module.conv.stride
    bias = module.conv.bias is not None
    g = module.conv.groups
    d = module.conv.dilation
    # 将矩阵形式的权重转换为卷积模块形式的权重
    if bias:
        compressed_weight = (decoding_matrix @ encoding_matrix)
        compressed_weight, compressed_bias = compressed_weight[:, :-1], compressed_weight[:, -1]
        compressed_module.eval().to(module.conv.weight.device)
        compressed_module.conv.load_state_dict({
            "weight": compressed_weight.reshape(c_out, c_in, k[0], k[1]).detach(),
            "bias": compressed_bias.reshape(c_out).detach()
        })
        # # 分离权重和偏置部分
        # encoding_matrix, encoding_bias = encoding_matrix[:, :-1], encoding_matrix[:, -1]
        # # 重新构造编码卷积层
        # compress_conv = nn.Conv2d(c_in, n_components_compressed, 
        #                           kernel_size=k[0], stride=s[0], 
        #                           padding=p[0], bias=True)
        
        # # 设置编码卷积层的权重和偏置
        # with torch.no_grad():
        #     # 权重形状: [n_components_compressed, in_channels, kernel_size, kernel_size]
        #     compress_conv.weight.data = encoding_matrix.reshape(n_components_compressed, c_in, 
        #                                                         k[0], k[1]).detach()
        #     # 偏置形状: [n_components_compressed]
        #     compress_conv.bias.data = encoding_bias.detach()
    else:
        compressed_weight = (decoding_matrix @ encoding_matrix)
        compressed_module.eval().to(module.conv.weight.device)
        compressed_module.conv.load_state_dict({
            "weight": compressed_weight.reshape(c_out, c_in, k[0], k[1]).detach()
        })
        # # 无偏置情况
        # # 重新构造编码卷积层
        # compress_conv = nn.Conv2d(c_in, n_components_compressed, 
        #                           kernel_size=k[0], stride=s[0], 
        #                           padding=p[0], bias=False)
        # # 设置编码卷积层的权重
        # with torch.no_grad():
        #     # 权重形状: [n_components_compressed, in_channels, kernel_size, kernel_size]
        #     compress_conv.weight.data = encoding_matrix.reshape(n_components_compressed, c_in, 
        #                                                         k[0], k[1]).detach()
    # # 构造解码卷积层(1x1卷积)
    # expand_conv = nn.Conv2d(n_components_compressed, c_out, 
    #                         kernel_size=1, stride=1, padding=0, bias=False)
    
    # # 设置解码卷积层的权重
    # with torch.no_grad():
    #     # 权重形状: [out_channels, n_components_compressed, 1, 1]
    #     expand_conv.weight.data = decoding_matrix.reshape(c_out, n_components_compressed, 1, 1)
    
    # # 返回级联的卷积模块
    # hourglass_conv = HourglassConv(c_in, n_components_compressed, c_out, 
    #                                k=k[0], s=s[0],
    #                                p=p[0], g=g, d=d[0]).eval().to(module.conv.weight.device)
    # hourglass_conv.bn.load_state_dict(module.bn.state_dict()) # copy batchnorm parameters
    # hourglass_conv.conv_compress.load_state_dict(compress_conv.state_dict()) # copy encoding conv parameters
    # hourglass_conv.conv_expand.load_state_dict(expand_conv.state_dict()) # copy decoding conv parameters
    
    # return hourglass_conv, n_components_compressed
    return n_components_compressed


def compress_model(base_model, compressed_model, pca_cache, ratio, layers):
    """
    压缩模型，用等效的级联卷积模块替换原始的卷积模块

    Args:
        base_model(ultralytics.nn.tasks.DetectionModel): 需要压缩的模型
        compressed_model(ultralytics.nn.tasks.DetectionModel): 压缩后的模型
        pca_cache(dict): 原始模型的PCA缓存
        ratio(float): 选择保留的PCA成分的累积方差比例
        layers(list): 选择进行压缩的层id列表
    """
    # 遍历模型模块列表，查找需要压缩的模块，用等效的级联卷积模块替换原始的卷积模块

    def _match(n, m, lid):
        return f"model.{lid}." in n and isinstance(m, Conv)
    
    for lid in layers:
        for name, module in base_model.named_modules():
            if _match(name, module, lid):
                # 获取压缩模型当中对应的模块
                compressed_module = compressed_model.get_submodule(name)
                
                # 获取PCA参数
                variances = pca_cache[name].explained_variance_.to(module.conv.weight.device)
                means = pca_cache[name].mean_.to(module.conv.weight.device)
                components = pca_cache[name].components_.to(module.conv.weight.device)

                # 计算压缩模块的权重，并加载到压缩模型当中
                n_components_compressed = compress_conv_module(module, compressed_module, variances, means, components, ratio)

                # 计算压缩前后的损失值
                c_in, k_size = module.conv.in_channels, module.conv.kernel_size
                c_hidden = n_components_compressed

                test_input = torch.randn(1, c_in, 64, 64).to(module.conv.weight.device)
                test_output = module(test_input)
                test_output_compressed = compressed_module(test_input)
                loss = F.mse_loss(test_output, test_output_compressed)
                LOGGER.info(f"Module {name} compression loss: {loss.item()}")
                LOGGER.info(f"Module {name} channel compression ratio: {(c_hidden / (c_in*k_size[0]*k_size[1]))*100:.2f}%")
    
    return compressed_model


def main(args):
    # 加载模型
    base_model = YOLO(args.base_model).eval().to("cuda")
    # 加载压缩后模型的架构配置文件
    model_cfg = yaml_model_load(args.model_cfg)
    model_cfg["nc"] = base_model.model.nc
    YAML.save(data=model_cfg, file=f"{args.save_path.split('.')[0]}.yaml")
    compressed_model = YOLO(f"{args.save_path.split('.')[0]}.yaml").eval().to("cuda")
    compressed_model.model.load_state_dict(base_model.model.state_dict()) # 先加载基础模型的权重，初始化非压缩模块
    # 加载PCA缓存
    LOGGER.info(f"Loading PCA cache from {args.pca_cache_path}")
    if os.path.exists(args.pca_cache_path):
        with open(args.pca_cache_path, "rb") as f:
            pca_cache = joblib.load(f)
    else:
        raise FileNotFoundError(f"PCA cache file {args.pca_cache_path} not found")
    # 压缩模型
    if args.layers is None:
        args.layers = list(range(len(base_model.model.model)))
    LOGGER.info(f"Compressing model with ratio {args.ratio} and layers {args.layers}")
    compress_model(base_model.model, compressed_model.model, pca_cache, args.ratio, args.layers) # 会将压缩后的模型权重加载到compressed_model.model当中
    # 计算压缩前后的损失值
    pbar = tqdm(os.listdir(args.sample_images)[:args.sample_num])
    total_loss = 0.0
    for i, image in enumerate(pbar):
        test_input = cv2.imread(os.path.join(args.sample_images, image))
        test_input = resize(to_tensor(test_input), (640, 640)).unsqueeze(0).to("cuda")
        test_output = base_model.model(test_input)
        test_output_compressed = compressed_model.model(test_input)
        loss = F.mse_loss(test_output[0], test_output_compressed[0])
        for j in range(len(test_output[1])):
            loss += F.mse_loss(test_output[1][j], test_output_compressed[1][j])
        loss /= len(test_output[1]) + 1.0
        total_loss += loss.item()
        pbar.set_description(f"Average compression loss: {total_loss / (i + 1):.4f}")
    # 保存模型
    compressed_model.save(args.save_path)
    LOGGER.info(f"Compressed model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--model_cfg", type=str, default=None, help="Target model config path(model cfg after compression)")
    parser.add_argument("--pca_cache_path", type=str, required=True, help="PCA cache path")
    parser.add_argument("--save_path", type=str, required=True, help="Save path")
    parser.add_argument("--ratio", type=float, required=True, help="Ratio")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Layers")
    parser.add_argument("--sample_images", type=str, default=None, help="Sample images path")
    parser.add_argument("--sample_num", type=int, default=100, help="Sample images number")
    args = parser.parse_args()
    main(args)