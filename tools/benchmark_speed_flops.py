"""Benchmark single-image inference latency and FLOPs for YOLO and Faster R-CNN.

Reports, for each model:
  - end-to-end latency per image (preprocess + forward + postprocess), mean/std
    over repeated runs after warmup, and FPS;
  - FLOPs and parameter count at a fixed input resolution.

Run from the myultralytics repo root inside the prepared conda environment, e.g.:

    python tools/benchmark_speed_flops.py \
        --yolo-weights yolov8n.pt \
        --frcnn-config ../NSGP-RePRE-main/work_dirs/naive/cl_faster_rcnn_naive_15_5_2/cl_faster_rcnn_naive_15_5_2.py \
        --frcnn-checkpoint ../NSGP-RePRE-main/work_dirs/naive/cl_faster_rcnn_naive_15_5_2/best_pascal_voc_mAP_epoch_14.pth

All paths are relative to the current working directory.
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Single-image latency + FLOPs benchmark for YOLO and Faster R-CNN')
    parser.add_argument('--yolo-weights', default='yolov8n.pt',
                        help='YOLO checkpoint, relative to cwd')
    parser.add_argument('--frcnn-config',
                        default='../NSGP-RePRE-main/work_dirs/naive/'
                                'cl_faster_rcnn_naive_15_5_2/cl_faster_rcnn_naive_15_5_2.py',
                        help='Faster R-CNN mmdet config, relative to cwd')
    parser.add_argument('--frcnn-checkpoint',
                        default='../NSGP-RePRE-main/work_dirs/naive/'
                                'cl_faster_rcnn_naive_15_5_2/best_pascal_voc_mAP_epoch_14.pth',
                        help='Faster R-CNN checkpoint, relative to cwd')
    parser.add_argument('--mmdet-root', default='../NSGP-RePRE-main',
                        help='mmdetection repo root (for sys.path), relative to cwd')
    parser.add_argument('--image',
                        default='data/VOC/VOCdevkit/VOC2007/JPEGImages/005770.jpg',
                        help='test image, relative to cwd')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='YOLO input resolution; FLOPs are computed on the '
                             'real preprocessed input for each model')
    parser.add_argument('--warmup', type=int, default=10,
                        help='warmup iterations before timing')
    parser.add_argument('--iters', type=int, default=50,
                        help='timed iterations')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def check_file(path_str, kind):
    """Fail fast if a required file is missing; return the Path."""
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f'{kind} not found: {path} (cwd={Path.cwd()})')
    return path


def time_inference(infer_fn, warmup, iters):
    """Time a zero-arg inference callable; returns (mean_ms, std_ms, fps).

    infer_fn must run the full single-image inference once per call.
    """
    for _ in range(warmup):
        infer_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times_ms = []
    for _ in range(iters):
        start = time.perf_counter()
        infer_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    times = np.asarray(times_ms)  # (iters,)
    mean_ms = float(times.mean())
    return mean_ms, float(times.std()), 1000.0 / mean_ms


def benchmark_yolo(args, image_path):
    """Return dict with latency stats, FLOPs (G) and params (M) for a YOLO model."""
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    yolo = YOLO(str(check_file(args.yolo_weights, 'YOLO weights')))
    yolo.to(args.device)
    yolo.model.eval()

    with torch.no_grad():
        infer_fn = lambda: yolo.predict(str(image_path), imgsz=args.imgsz,
                                        device=args.device, verbose=False)
        mean_ms, std_ms, fps = time_inference(infer_fn, warmup=args.warmup, iters=args.iters)

    with torch.no_grad():
        flops = get_flops(yolo.model, imgsz=args.imgsz)  # GFLOPs at (1, 3, imgsz, imgsz)
    n_params = sum(p.numel() for p in yolo.model.parameters()) / 1e6
    return dict(mean_ms=mean_ms, std_ms=std_ms, fps=fps,
                gflops=float(flops), mparams=n_params)


def benchmark_frcnn(args, image_path):
    """Return dict with latency stats, FLOPs (G) and params (M) for Faster R-CNN."""
    mmdet_root = Path(args.mmdet_root)
    if not (mmdet_root / 'mmdet' / 'version.py').is_file():
        raise FileNotFoundError(
            f'mmdetection repo not found at {mmdet_root} (expected mmdet/version.py inside)')
    sys.path.insert(0, str(mmdet_root.resolve()))

    from mmengine.analysis import get_model_complexity_info
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet.apis import inference_detector, init_detector
    from mmdet.utils import get_test_pipeline_cfg

    config_path = check_file(args.frcnn_config, 'Faster R-CNN config')
    ckpt_path = check_file(args.frcnn_checkpoint, 'Faster R-CNN checkpoint')
    cfg = Config.fromfile(str(config_path))
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    model = init_detector(str(config_path), str(ckpt_path), device=args.device)
    model.eval()

    # Latency: full single-image pipeline (preprocess + forward + postprocess).
    with torch.no_grad():
        infer_fn = lambda: inference_detector(model, str(image_path))
        mean_ms, std_ms, fps = time_inference(infer_fn, warmup=args.warmup, iters=args.iters)

    # FLOPs: run mmengine's complexity analysis on the real preprocessed input.
    # forward is partially bound to data_samples so the profiler only sees the
    # input tensor, mirroring mmdet's tools/analysis_tools/get_flops.py.
    pipeline = Compose(get_test_pipeline_cfg(cfg))
    data = pipeline(dict(img_path=str(image_path), img_id=0))
    data = pseudo_collate([data])
    with torch.no_grad():
        data = model.data_preprocessor(data, False)
        _forward = model.forward
        model.forward = partial(_forward, data_samples=data['data_samples'])
        outputs = get_model_complexity_info(
            model, None, inputs=data['inputs'],  # (1, 3, H_pad, W_pad)
            show_table=False, show_arch=False)
        model.forward = _forward
    return dict(mean_ms=mean_ms, std_ms=std_ms, fps=fps,
                gflops=float(outputs['flops']) / 1e9,
                mparams=float(outputs['params']) / 1e6)


def print_result(name, stats):
    print(f'\n=== {name} ===')
    print(f'  latency : {stats["mean_ms"]:.2f} +/- {stats["std_ms"]:.2f} ms/image')
    print(f'  fps     : {stats["fps"]:.2f}')
    print(f'  FLOPs   : {stats["gflops"]:.2f} G')
    print(f'  params  : {stats["mparams"]:.2f} M')


def main():
    args = parse_args()
    image_path = check_file(args.image, 'test image')

    print(f'device={args.device}  image={image_path}  '
          f'warmup={args.warmup}  iters={args.iters}  imgsz={args.imgsz}')

    yolo_stats = benchmark_yolo(args, image_path)
    print_result(f'YOLO ({args.yolo_weights})', yolo_stats)

    frcnn_stats = benchmark_frcnn(args, image_path)
    print_result(f'Faster R-CNN ({Path(args.frcnn_config).name})', frcnn_stats)


if __name__ == '__main__':
    main()
