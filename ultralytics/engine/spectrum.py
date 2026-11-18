"""Eigen-spectrum utilities shared by the ESPReg and NSGP regularizers."""

import numpy as np
import scipy.ndimage
import torch


def find_elbow_point(svals: torch.Tensor) -> int:
    """Find the elbow point in a descending singular-value spectrum.

    The location is the maximum second-order drop of the smoothed spectrum,
    matching the adaptive threshold of the reference NSGP implementation.

    Args:
        svals: Singular values in descending order, shape (N,).

    Returns:
        int: Index of the elbow point.
    """
    points = svals.cpu().numpy()
    assert points.ndim == 1

    if len(points) >= 128:
        fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
        _delta = 1
        diff_o1 = fil_points[:-_delta] - fil_points[_delta:]
        diff_o2 = diff_o1[:-1] - diff_o1[1:]
        _drop_ratio = 0.03
        drop_num = int(len(points) * _drop_ratio / 2)
        assert len(points) - drop_num >= 10
        valid_o2 = diff_o2[drop_num:-drop_num]
        thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
    else:
        diff_o1 = points[:-1] - points[1:]
        diff_o2 = diff_o1[:-1] - diff_o1[1:]
        thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]

    i_thres = np.arange(len(points))[points >= thres_val].max()
    return int(i_thres)
