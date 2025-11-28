import argparse
import warnings
import joblib

import torch
from torch import nn

from ultralytics import YOLO

from tools.pca import PCAHooker, PCAHookerWithBboxes


MILESTONES = [0.75, 0.90, 0.95, 0.99]
MILESTONE_COLORS = ['red', 'orange', 'green', 'purple']


def main(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        warnings.warn("CUDA is not available, using CPU instead.")
    
    base_model = YOLO(args.base_model)
    incremental_model = YOLO(args.incremental_model)
    base_model.to(args.device).eval()
    incremental_model.to(args.device).eval()

    if args.layers is None:
        args.layers = list(range(len(base_model.model.model)))
    
    kernel_updates = {}

    if args.label_dir is not None:
        pca_hookers = PCAHookerWithBboxes(base_model, args.layers, modules=None, bboxes=None, device=args.device, check=False, unfold=True)
    else:
        pca_hookers = PCAHooker(base_model, args.layers, modules=None, device=args.device, check=False, unfold=True)
    
    pca_hookers.load_pca_cache(args.pca_cache)
    
    for name, module in base_model.model.named_modules():
        for layer in args.layers:
            if f"model.{layer}" in name and isinstance(nn.Conv2d):
                k, g, c_in, c_out = module.kernel_size, module.groups, module.in_channels, module.out_channels

                base_kernel = module.weight.data.reshape(g, c_out//g, c_in//g * k[0] * k[1])  # [g, c_out//g, c_in//g * k[0] * k[1]]
                
                incremental_module = incremental_model.model.get_submodule(name)
                incremental_kernel = incremental_module.weight.data.reshape(g, c_out//g, c_in//g * k[0] * k[1])  # [g, c_out//g, c_in//g * k[0] * k[1]]

                kernel_updates[name] = incremental_kernel - base_kernel

                break

    for name in kernel_updates.keys():
        if name in pca_hookers.names:
            for ig in range(kernel_updates[name].shape[0]):
                components, variances = pca_hookers.get_pca_results(name, ig)
                projections = torch.mean(kernel_updates @ components.T, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--incremental_model", type=str, required=True)
    parser.add_argument("--pca_cache", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=str, default=None)
    args = parser.parse_args()
    
    main(args)