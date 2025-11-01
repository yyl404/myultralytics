import os
import joblib
import argparse
from copy import deepcopy
from tqdm import tqdm
import cv2

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, resize

from ultralytics import YOLO
from ultralytics.nn.modules import Conv, Bottleneck, C2f, SPPF, Detect, \
    DecomposeConv, BottleneckDecomposed, C2fDecomposed, SPPFDecomposed, DetectDecomposed
from ultralytics.utils import LOGGER, YAML
from ultralytics.nn.tasks import yaml_model_load


def compress_conv_module(conv, name, pca_operators, ratio, linear_first=True, verbose=False):
    """
    压缩卷积模块

    Args:
        conv(nn.Conv2d): 需要压缩的卷积模块
        pca_operators(list): PCA算子列表（每个分组对应一个PCA算子）
        ratio(float): 选择保留的PCA成分的累积方差比例
        linear_first(bool): 是否先进行1x1卷积
        verbose(bool): 是否打印详细信息
    Returns:
        tuple: (conv_1, conv_2)
    """
    assert isinstance(conv, nn.Conv2d), "conv must be a nn.Conv2d module"
    assert conv.groups == len(pca_operators), "Group number of convolution module must be equal to the number of PCA operators"
    
    groups = conv.groups
    in_channels = conv.in_channels
    out_channels = conv.out_channels
    kernel_size = conv.kernel_size[0]
    stride = conv.stride[0]
    padding = conv.padding[0]
    dilation = conv.dilation[0]
    bias = conv.bias is not None

    weight = conv.weight.data.reshape(groups, out_channels//groups, in_channels//groups, kernel_size, kernel_size) # [groups, out_channels//groups, in_channels//groups, kernel_size, kernel_size]
    cr = []
    for ig in range(conv.groups):
        pca_operator = pca_operators[ig]
        variances = pca_operator.explained_variance_
        cumsum_variances = torch.cumsum(variances, dim=0)
        cumsum_variances_normalized = cumsum_variances / cumsum_variances[-1]
        for i in range(len(cumsum_variances_normalized)):
            if cumsum_variances_normalized[i] >= ratio:
                cr.append(i)
                break
    if len(cr) == 0:
        cr = len(cumsum_variances_normalized)
    else:
        cr = max(cr)

    conv_1_weight = []
    conv_2_weight = []
    if linear_first:
        for ig in range(groups):
            pca_operator = pca_operators[ig]
            components = pca_operator.components_
            components_kept = components[:cr]
            
            weight_grouped = weight[ig] # [out_channels//groups, in_channels//groups, kernel_size, kernel_size]
        
            conv_1_weight_grouped = components_kept[:,:,None,None] # [cr, in_channels//groups, 1, 1]
            weight_permuted = weight_grouped.permute(2, 3, 0, 1) # [kernel_size, kernel_size, out_channels//groups, in_channels//groups]
            conv_2_weight_grouped = weight_permuted @ components_kept.T # [kernel_size, kernel_size, out_channels//groups, in_channels//groups] @ [in_channels//groups, cr] --> [kernel_size, kernel_size, out_channels//groups, cr]
            conv_2_weight_grouped = conv_2_weight_grouped.permute(2, 3, 0, 1) # [out_channels//groups, cr, kernel_size, kernel_size]
            conv_1_weight.append(conv_1_weight_grouped)
            conv_2_weight.append(conv_2_weight_grouped)
        conv_1_weight = torch.cat(conv_1_weight, dim=0) # [groups*cr, in_channels//groups, 1, 1]
        conv_2_weight = torch.cat(conv_2_weight, dim=0) # [out_channels, cr, kernel_size, kernel_size]
        conv_1 = nn.Conv2d(in_channels, cr, 1, groups=groups, bias=False)
        conv_1.weight.data = conv_1_weight
        conv_2 = nn.Conv2d(cr, out_channels, kernel_size, stride, padding, groups=groups, dilation=dilation, bias=False)
        conv_2.weight.data = conv_2_weight
    else:
        for ig in range(groups):
            pca_operator = pca_operators[ig]
            components = pca_operator.components_
            components_kept = components[:cr]
            
            weight_grouped = weight[ig] # [out_channels//groups, in_channels//groups, kernel_size, kernel_size]

            conv_1_weight_grouped = components_kept[:,:,None,None] # [cr, in_channels//groups*kernel_size*kernel_size]
            conv_1_weight_grouped = conv_1_weight_grouped.reshape(cr, in_channels//groups, kernel_size, kernel_size) # [cr, in_channels//groups, kernel_size, kernel_size]
            conv_2_weight_grouped = weight_grouped.reshape(out_channels//groups, in_channels//groups*kernel_size*kernel_size) # [out_channels//groups, in_channels//groups*kernel_size*kernel_size]
            conv_2_weight_grouped = conv_2_weight_grouped @ components_kept.T # [out_channels//groups, in_channels//groups*kernel_size*kernel_size] @ [in_channels//groups*kernel_size*kernel_size, cr] --> [out_channels//groups, cr]
            conv_2_weight_grouped = conv_2_weight_grouped.reshape(out_channels//groups, cr, 1, 1) # [out_channels//groups, cr, 1, 1]
            conv_1_weight.append(conv_1_weight_grouped)
            conv_2_weight.append(conv_2_weight_grouped)

        conv_1_weight = torch.cat(conv_1_weight, dim=0) # [groups*cr, in_channels//groups, kernel_size, kernel_size]
        conv_2_weight = torch.cat(conv_2_weight, dim=0) # [out_channels, cr, 1, 1]
        conv_1 = nn.Conv2d(in_channels, cr, kernel_size, stride, padding, groups=groups, dilation=dilation, bias=False)
        conv_1.eval().to(conv.weight.device)
        conv_1.weight.data = conv_1_weight
        conv_2 = nn.Conv2d(cr, out_channels, 1, groups=groups, bias=False)
        conv_2.eval().to(conv.weight.device)
        conv_2.weight.data = conv_2_weight
    
    if bias:
        conv_2.register_parameter("bias", nn.Parameter(conv.bias.data))

    if verbose:
        test_input = torch.randn(1, in_channels, 64, 64).to(conv.weight.device)
        test_output = conv(test_input)
        test_output_compressed = conv_2(conv_1(test_input))
        loss = F.mse_loss(test_output, test_output_compressed)

        compression_ratio = conv_1.out_channels/(conv_1.in_channels*conv_1.kernel_size[0]*conv_1.kernel_size[0])
        LOGGER.info(f"Conv module {name} compression loss: {loss.item():.6f}, compression ratio: {compression_ratio*100:.2f}%")
    return conv_1, conv_2
            
# -------- Non-recursive helpers to reduce analyzer complexity --------
def _compress_conv(module: Conv, name, pca_cache, ratio, linear_first, verbose):
    conv = module.conv
    conv_name = name + ".conv"
    pca_operators = pca_cache[conv_name]
    conv_1, conv_2 = compress_conv_module(conv, name, pca_operators, ratio, linear_first, verbose)
    cr = conv_1.out_channels
    compressed_module = DecomposeConv(conv.in_channels, conv.out_channels, conv.kernel_size[0],
                                      conv.stride[0], conv.padding[0], conv.groups, conv.dilation[0],
                                      cr=cr, linear_first=linear_first)
    compressed_module.eval().to(conv.weight.device)
    compressed_module.conv_1.load_state_dict(conv_1.state_dict())
    compressed_module.conv_2.load_state_dict(conv_2.state_dict())
    compressed_module.bn.load_state_dict(module.bn.state_dict())
    decomposition_args = {
        "cr": cr,
        "linear_first": linear_first
    }
    return compressed_module, decomposition_args


def _compress_bottleneck(module: Bottleneck, name, pca_cache, ratio, linear_first, verbose):
    cv1 = module.cv1
    cv2 = module.cv2
    shortcut = module.add
    g = module.cv2.conv.groups
    k = (module.cv1.conv.kernel_size[0], module.cv2.conv.kernel_size[0])
    e = module.cv1.conv.out_channels / module.cv2.conv.out_channels + 1e-3

    cv1_compressed, decomposition_args_cv1 = _compress_conv(cv1, name + ".cv1", pca_cache, ratio, linear_first, verbose)
    cv2_compressed, decomposition_args_cv2 = _compress_conv(cv2, name + ".cv2", pca_cache, ratio, linear_first, verbose)

    decomposition_args = {
        "cr1": decomposition_args_cv1["cr"],
        "cr2": decomposition_args_cv2["cr"],
        "linear_first": linear_first
    }
    compressed_module = BottleneckDecomposed(cv1.conv.in_channels, cv2.conv.out_channels, shortcut=shortcut, g=g, k=k, e=e, decomposition_args=decomposition_args)
    compressed_module.eval().to(cv1.conv.weight.device)
    compressed_module.cv1.load_state_dict(cv1_compressed.state_dict())
    compressed_module.cv2.load_state_dict(cv2_compressed.state_dict())
    return compressed_module, decomposition_args


def _compress_modulelist(module: nn.ModuleList, name, pca_cache, ratio, linear_first, verbose):
    compressed_module = nn.ModuleList()
    decomposition_args = []
    for i, m in enumerate(module):
        # elements are Bottleneck in this architecture
        compressed_module_item, decomposition_args_item = _compress_bottleneck(m, name + "." + str(i), pca_cache, ratio, linear_first, verbose)
        compressed_module.append(compressed_module_item)
        decomposition_args.append((decomposition_args_item["cr1"], decomposition_args_item["cr2"]))
    return compressed_module, decomposition_args


def _compress_c2f(module: C2f, name, pca_cache, ratio, linear_first, verbose):
    cv1 = module.cv1
    cv2 = module.cv2
    m = module.m
    shortcut = m[0].add
    g = m[0].cv2.conv.groups
    e = module.c / module.cv2.conv.out_channels + 1e-3

    cv1_compressed, decomposition_args_cv1 = _compress_conv(cv1, name + ".cv1", pca_cache, ratio, linear_first, verbose)
    cv2_compressed, decomposition_args_cv2 = _compress_conv(cv2, name + ".cv2", pca_cache, ratio, linear_first, verbose)
    m_compressed, decomposition_args_m = _compress_modulelist(m, name + ".m", pca_cache, ratio, linear_first, verbose)

    decomposition_args = {
        "cr1": decomposition_args_cv1["cr"],
        "cr2": decomposition_args_cv2["cr"],
        "cr_bottleneck": decomposition_args_m,
        "linear_first": linear_first
    }
    compressed_module = C2fDecomposed(cv1.conv.in_channels, cv2.conv.out_channels, n=len(m), shortcut=shortcut, g=g, e=e, decomposition_args=decomposition_args)
    compressed_module.eval().to(cv1.conv.weight.device)
    compressed_module.cv1.load_state_dict(cv1_compressed.state_dict())
    compressed_module.cv2.load_state_dict(cv2_compressed.state_dict())
    compressed_module.m.load_state_dict(m_compressed.state_dict())
    return compressed_module, decomposition_args


def _compress_sppf(module: SPPF, name, pca_cache, ratio, linear_first, verbose):
    cv1 = module.cv1
    cv2 = module.cv2
    k = module.m.kernel_size[0]
    cv1_compressed, decomposition_args_cv1 = _compress_conv(cv1, name + ".cv1", pca_cache, ratio, linear_first, verbose)
    cv2_compressed, decomposition_args_cv2 = _compress_conv(cv2, name + ".cv2", pca_cache, ratio, linear_first, verbose)
    decomposition_args = {
        "cr1": decomposition_args_cv1["cr"],
        "cr2": decomposition_args_cv2["cr"],
        "linear_first": linear_first
    }
    compressed_module = SPPFDecomposed(cv1.conv.in_channels, cv2.conv.out_channels, k=k, decomposition_args=decomposition_args)
    compressed_module.eval().to(cv1.conv.weight.device)
    compressed_module.cv1.load_state_dict(cv1_compressed.state_dict())
    compressed_module.cv2.load_state_dict(cv2_compressed.state_dict())
    return compressed_module, decomposition_args


def _compress_detect(module: Detect, name, pca_cache, ratio, linear_first, verbose):
    cv2 = module.cv2
    cv3 = module.cv3
    nc = module.nc
    nl = module.nl
    ch = []

    cv2_compressed = []
    cv3_compressed = []
    decomposition_args = {}

    for i in range(nl):
        cv2_compressed.append([])
        cv3_compressed.append([])
        ch.append(cv2[i][0].conv.in_channels)
        for j in range(len(cv2[i])):
            m = cv2[i][j]
            if isinstance(m, Conv):
                m_compressed, decomposition_args_m = _compress_conv(m, name + ".cv2." + str(i) + "." + str(j), pca_cache, ratio, linear_first, verbose)
                cv2_compressed[i].append(m_compressed)
                decomposition_args[f"cr_cv2_{j+1}_{i+1}"] = decomposition_args_m["cr"]
            elif isinstance(m, nn.Conv2d):
                conv_1, conv_2 = compress_conv_module(m, name + ".cv2." + str(i) + "." + str(j), pca_cache[name + ".cv2." + str(i) + "." + str(j)], ratio, linear_first, verbose)
                cv2_compressed[i].append(conv_1)
                cv2_compressed[i].append(conv_2)
                decomposition_args[f"cr_cv2_{j+1}_{i+1}"] = conv_1.out_channels

            m = cv3[i][j]
            if isinstance(m, Conv):
                m_compressed, decomposition_args_m = _compress_conv(m, name + ".cv3." + str(i) + "." + str(j), pca_cache, ratio, linear_first, verbose)
                cv3_compressed[i].append(m_compressed)
                decomposition_args[f"cr_cv3_{j+1}_{i+1}"] = decomposition_args_m["cr"]
            elif isinstance(m, nn.Conv2d):
                conv_1, conv_2 = compress_conv_module(m, name + ".cv3." + str(i) + "." + str(j), pca_cache[name + ".cv3." + str(i) + "." + str(j)], ratio, linear_first, verbose)
                cv3_compressed[i].append(conv_1)
                cv3_compressed[i].append(conv_2)
                decomposition_args[f"cr_cv3_{j+1}_{i+1}"] = conv_1.out_channels
        cv2_compressed[i] = nn.Sequential(*cv2_compressed[i])
        cv3_compressed[i] = nn.Sequential(*cv3_compressed[i])
    cv2_compressed = nn.ModuleList(cv2_compressed)
    cv3_compressed = nn.ModuleList(cv3_compressed)
    decomposition_args["linear_first"] = linear_first

    compressed_module = DetectDecomposed(nc, ch, decomposition_args=decomposition_args)
    compressed_module.eval().to(cv2[0][0].conv.weight.device)

    compressed_module.cv2.load_state_dict(cv2_compressed.state_dict())
    compressed_module.cv3.load_state_dict(cv3_compressed.state_dict())
    compressed_module.dfl.load_state_dict(module.dfl.state_dict())

    compressed_module.stride = module.stride
    return compressed_module, decomposition_args

def compress_module(module, name, pca_cache, ratio, linear_first=True, verbose=False):
    """
    压缩模块，返回近似等效的压缩模块

    Args:
        module(Conv): 需要压缩的模块
        name(str): 模块名称
        pca_cache(dict): PCA缓存数据
        ratio(float): 选择保留的PCA成分的累积方差比例
        linear_first(bool): 是否先进行1x1卷积
        verbose(bool): 是否打印详细信息
    Returns:
        tuple: (compressed_module, decomposition_args)
    """
    if isinstance(module, Conv):
        compressed_module, decomposition_args = _compress_conv(module, name, pca_cache, ratio, linear_first, verbose)
    elif isinstance(module, Bottleneck):
        compressed_module, decomposition_args = _compress_bottleneck(module, name, pca_cache, ratio, linear_first, verbose)
    elif isinstance(module, nn.ModuleList):
        compressed_module, decomposition_args = _compress_modulelist(module, name, pca_cache, ratio, linear_first, verbose)
    elif isinstance(module, C2f):
        compressed_module, decomposition_args = _compress_c2f(module, name, pca_cache, ratio, linear_first, verbose)
    elif isinstance(module, SPPF):
        compressed_module, decomposition_args = _compress_sppf(module, name, pca_cache, ratio, linear_first, verbose)
    elif isinstance(module, Detect):
        compressed_module, decomposition_args = _compress_detect(module, name, pca_cache, ratio, linear_first, verbose)
    else:
        raise ValueError(f"Unsupported module type: {type(module).__name__}")
    
    if hasattr(module, "i"):
        compressed_module.np = sum(x.numel() for x in compressed_module.parameters())  # number params
        compressed_module.i, compressed_module.f = module.i, module.f  # attach index, 'from' index
        compressed_module.type = str(compressed_module)[8:-2].replace("__main__.", "")  # module type
    
    if verbose:
        if isinstance(module, Conv):
            test_input = torch.randn(1, module.conv.in_channels, 64, 64).to(module.conv.weight.device)
        elif isinstance(module, Bottleneck):
            test_input = torch.randn(1, module.cv1.conv.in_channels, 64, 64).to(module.cv1.conv.weight.device)
        elif isinstance(module, nn.ModuleList):
            test_input = torch.randn(1, module[0].cv1.conv.in_channels, 64, 64).to(module[0].cv1.conv.weight.device)
        elif isinstance(module, C2f):
            test_input = torch.randn(1, module.cv1.conv.in_channels, 64, 64).to(module.cv1.conv.weight.device)
        elif isinstance(module, SPPF):
            test_input = torch.randn(1, module.cv1.conv.in_channels, 64, 64).to(module.cv1.conv.weight.device)
        elif isinstance(module, Detect):
            test_input = [torch.randn(1, c, 64, 64).to(module.cv2[0][0].conv.weight.device) for c in [module.cv2[i][0].conv.in_channels for i in range(module.nl)]]
        
        if isinstance(test_input, list):
            test_output = module(deepcopy(test_input))
            test_output_compressed = compressed_module(test_input)
            loss = []
            for i in range(len(test_output)):
                for j in range(len(test_output[i])):
                    loss.append(F.mse_loss(test_output[i][j], test_output_compressed[i][j]))
            loss = sum(loss) / len(loss)
        elif isinstance(module, nn.ModuleList):
            test_output = test_input
            test_output_compressed = test_input
            loss = []
            for i in range(len(module)):
                test_output = module[i](test_output)
                test_output_compressed = compressed_module[i](test_output_compressed)
                loss.append(F.mse_loss(test_output, test_output_compressed))
            loss = sum(loss) / len(loss)
        else:
            test_output = module(test_input)
            test_output_compressed = compressed_module(test_input)
            loss = F.mse_loss(test_output, test_output_compressed)
        LOGGER.info(f"Module {name} compression loss: {loss.item():.6f}, type: {type(compressed_module).__name__}")

    return compressed_module, decomposition_args

def compress_model(base_model, base_model_cfg, pca_cache, ratio, layers, linear_first=True, verbose=False):
    """
    压缩模型，返回压缩后的模型和架构配置文件

    Args:
        base_model(ultralytics.nn.tasks.DetectionModel): 需要压缩的模型
        base_model_cfg(dict): 原始模型的架构配置文件
        pca_cache(dict): 原始模型的PCA缓存
        ratio(float): 选择保留的PCA成分的累积方差比例
        layers(list): 选择进行压缩的层id列表
        linear_first(bool): 是否先进行1x1卷积
        verbose(bool): 是否打印详细信息
    """
    compressed_model = deepcopy(base_model)
    compressed_model.model.eval()
    compressed_model_cfg = deepcopy(base_model_cfg)

    len_backbone = len(compressed_model_cfg["backbone"])
    len_head = len(compressed_model_cfg["head"])
    
    for lid in layers:
        name = f"model.{lid}"
        base_module = base_model.model.get_submodule(name)
        if type(base_module) in [Conv, C2f, SPPF, Detect]:
            # 计算压缩模块的权重
            compressed_module, decomposition_args = compress_module(base_module, name, pca_cache, ratio, linear_first, verbose)

            # 替换原始模块为压缩后的模块
            parent_module = compressed_model.model.get_submodule('.'.join(name.split('.')[:-1]) if '.' in name else '')
            parent_module.add_module(name.split('.')[-1], compressed_module)

            # 更新架构配置文件
            if lid < len_backbone:
                compressed_model_cfg["backbone"][lid][3].append(decomposition_args)
                compressed_model_cfg["backbone"][lid][2] = type(compressed_module).__name__
            else:
                compressed_model_cfg["head"][lid - len_backbone][3].append(decomposition_args)
                compressed_model_cfg["head"][lid - len_backbone][2] = type(compressed_module).__name__
    
    compressed_model.yaml = compressed_model_cfg
    return compressed_model, compressed_model_cfg


def main(args):
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        LOGGER.warning("CUDA is not available, using CPU")
    else:
        LOGGER.info(f"Using device {args.device}")
    # 加载模型
    base_model = YOLO(args.base_model).eval().to(args.device)
    # 加载原始模型的架构配置文件
    base_model_cfg = yaml_model_load(args.base_model_cfg)
    base_model_cfg["nc"] = base_model.model.nc
    # 加载PCA缓存
    LOGGER.info(f"Loading PCA cache from {args.pca_cache_path}")
    if os.path.exists(args.pca_cache_path):
        with open(args.pca_cache_path, "rb") as f:
            pca_cache = joblib.load(f)
    else:
        raise FileNotFoundError(f"PCA cache file {args.pca_cache_path} not found")
    # 压缩模型
    if args.layers is None:
        # 如果未指定层，则压缩所有层
        args.layers = list(range(len(base_model.model.model)))
    LOGGER.info(f"Compressing model with variance cumulative ratio {args.ratio}. Compressed layers: {args.layers}")
    compressed_model, compressed_model_cfg = compress_model(base_model, base_model_cfg, pca_cache, args.ratio, args.layers, args.linear_first, args.verbose) # 自动创建压缩后的模型和架构配置文件
    # 评估压缩后的模型性能
    pbar = tqdm(os.listdir(args.sample_images)[:args.sample_num])
    total_loss = 0.0
    for i, image in enumerate(pbar):
        test_input = cv2.imread(os.path.join(args.sample_images, image))
        test_input = resize(to_tensor(test_input), (640, 640)).unsqueeze(0).to(args.device)
        test_output = base_model.model(test_input)
        test_output_compressed = compressed_model.model(test_input)
        loss_box = 0.
        loss_feat = 0.
        loss_box += F.mse_loss(test_output[0], test_output_compressed[0])
        for j in range(len(test_output[1])):
            loss_feat += F.mse_loss(test_output[1][j], test_output_compressed[1][j])
        pbar.set_description(f"Average compression loss: {loss_box / (i + 1):.6f}(box), {loss_feat / (i + 1):.6f}(feat)")
    # 保存模型和架构配置文件
    state_dict_compressed_model = compressed_model.model.state_dict()
    YAML.save(data=compressed_model_cfg, file=args.save_path.split('.')[0] + ".yaml")
    compressed_model = YOLO(args.save_path.split('.')[0] + ".yaml")
    compressed_model.model.load_state_dict(state_dict_compressed_model)
    compressed_model.save(args.save_path)
    print(compressed_model.yaml)
    LOGGER.info(f"Compressed model and config saved to {args.save_path} and {args.save_path.split('.')[0] + '.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--base_model_cfg", type=str, default=None, help="Base model config path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--pca_cache_path", type=str, required=True, help="PCA cache path")
    parser.add_argument("--save_path", type=str, required=True, help="Save path")
    parser.add_argument("--ratio", type=float, required=True, help="Ratio")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Layers")
    parser.add_argument("--sample_images", type=str, default=None, help="Sample images path")
    parser.add_argument("--sample_num", type=int, default=100, help="Sample images number")
    parser.add_argument("--linear_first", action="store_true", help="Whether to compress the model with linear first")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose information")
    args = parser.parse_args()
    main(args)