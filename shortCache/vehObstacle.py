'''死车判定使用的路径几何计算。'''

from typing import Any

import numpy as np


def remaining_path_length(path_data: Any, current_position: Any) -> float:
    '''计算当前位置投影到path后，从投影点到path末端的弧长。'''
    try:
        path = np.asarray(path_data, dtype=float)
        position = np.asarray(current_position, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return 0.0
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] < 2 or position.size < 2:
        return 0.0

    points = path[:, :2]
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid_indices = np.flatnonzero(lengths > 1e-9)
    if valid_indices.size == 0:
        return 0.0

    valid_segments = segments[valid_indices]
    starts = points[valid_indices]
    ratios = np.clip(
        np.einsum('ij,ij->i', position[:2] - starts, valid_segments)
        / np.square(lengths[valid_indices]),
        0.0,
        1.0,
    )
    projections = starts + ratios[:, None] * valid_segments
    nearest = int(np.argmin(np.linalg.norm(projections - position[:2], axis=1)))
    segment_index = int(valid_indices[nearest])
    return float(
        (1.0 - ratios[nearest]) * lengths[segment_index]
        + lengths[segment_index + 1:].sum()
    )
