"""空间规划共用的二维折线几何。"""

import math

import numpy as np


def navigation_heading_to_cartesian(value):
    '''将正北为0、顺时针增加的接口航向转换为平面直角坐标系航向。'''
    return (0.5 * math.pi - float(value)) % (2.0 * math.pi)


def cartesian_heading_to_navigation(value):
    '''将平面直角坐标系航向转换为正北为0、顺时针增加的接口航向。'''
    return (0.5 * math.pi - float(value)) % (2.0 * math.pi)


def clean_xy(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    points = points[:, :2]
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        return points.copy()
    keep = np.concatenate(([True], np.linalg.norm(
        np.diff(points, axis=0), axis=1) > 1e-9))
    return points[keep].copy()


def arc_lengths(line):
    points = clean_xy(line)
    if len(points) == 0:
        return np.empty(0, dtype=float)
    return _arc_from_clean(points)


def polyline_length(line):
    values = arc_lengths(line)
    return float(values[-1]) if len(values) else 0.0


def point_at_s(line, target_s):
    points = clean_xy(line)
    if len(points) == 1:
        return points[0].copy()
    arc = _arc_from_clean(points)
    return _points_at_s_clean(points, arc, [target_s])[0]


def tangent_at_s(line, target_s):
    points = clean_xy(line)
    if len(points) < 2:
        return np.asarray([1.0, 0.0])
    arc = _arc_from_clean(points)
    target = float(np.clip(target_s, 0.0, arc[-1]))
    index = min(
        int(np.searchsorted(arc, target, side="right") - 1),
        len(points) - 2,
    )
    tangent = points[index + 1] - points[index]
    length = float(np.linalg.norm(tangent))
    return tangent / length if length > 1e-12 else np.asarray([1.0, 0.0])


def heading_at_s(line, target_s):
    tangent = tangent_at_s(line, target_s)
    return float(math.atan2(tangent[1], tangent[0]))


def resample_polyline(line, spacing):
    points = clean_xy(line)
    if len(points) < 2:
        return points
    arc = _arc_from_clean(points)
    total = float(arc[-1])
    sample_s = np.arange(0.0, total, max(float(spacing), 1e-3))
    if len(sample_s) == 0 or total - sample_s[-1] > 1e-9:
        sample_s = np.append(sample_s, total)
    else:
        sample_s[-1] = total
    return _points_at_s_clean(points, arc, sample_s)


def slice_polyline(line, start_s, end_s):
    points = clean_xy(line)
    if len(points) < 2:
        return points
    arc = _arc_from_clean(points)
    start = float(np.clip(start_s, 0.0, arc[-1]))
    end = float(np.clip(end_s, start, arc[-1]))
    boundaries = _points_at_s_clean(points, arc, [start, end])
    output = [boundaries[0]]
    output.extend(
        points[index].copy()
        for index in range(1, len(points) - 1)
        if start + 1e-9 < arc[index] < end - 1e-9
    )
    if end > start + 1e-9:
        output.append(boundaries[1])
    return clean_xy(np.asarray(output))


def truncate_polyline(line, end_s):
    return slice_polyline(line, 0.0, end_s)


def append_polyline(first, second):
    left = clean_xy(first)
    right = clean_xy(second)
    if len(left) == 0:
        return right
    if len(right) == 0:
        return left
    if np.linalg.norm(left[-1] - right[0]) <= 1e-8:
        right = right[1:]
    return clean_xy(np.vstack((left, right)))


def project_to_polyline(point, line):
    query = np.asarray(point, dtype=float)[:2]
    points = clean_xy(line)
    segments = np.diff(points, axis=0)
    lengths_squared = np.sum(segments * segments, axis=1)
    ratios = np.divide(
        np.sum((query - points[:-1]) * segments, axis=1),
        lengths_squared,
        out=np.zeros(len(segments), dtype=float),
        where=lengths_squared > 1e-12,
    )
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = points[:-1] + ratios[:, None] * segments
    distances = np.linalg.norm(projections - query, axis=1)
    index = int(np.argmin(distances))
    segment_length = math.sqrt(max(lengths_squared[index], 0.0))
    tangent = (
        segments[index] / segment_length
        if segment_length > 1e-12 else np.asarray([1.0, 0.0])
    )
    residual = query - projections[index]
    signed_lateral = tangent[0] * residual[1] - tangent[1] * residual[0]
    arc = _arc_from_clean(points)
    return {
        "s": float(arc[index] + ratios[index] * segment_length),
        "d": float(signed_lateral),
        "distance": float(distances[index]),
        "point": projections[index],
        "tangent": tangent,
        "segmentIndex": index,
    }


def project_points_to_polyline(query_points, line):
    """批量投影多个点，避免为每个点重复清理参考线和计算累计弧长。"""
    queries = clean_xy(query_points)
    points = clean_xy(line)
    segments = np.diff(points, axis=0)
    lengths_squared = np.sum(segments * segments, axis=1)
    residuals = queries[:, None, :] - points[:-1][None, :, :]
    ratios = np.divide(
        np.sum(residuals * segments[None, :, :], axis=2),
        lengths_squared[None, :],
        out=np.zeros((len(queries), len(segments)), dtype=float),
        where=lengths_squared[None, :] > 1e-12,
    )
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = (
        points[:-1][None, :, :]
        + ratios[:, :, None] * segments[None, :, :]
    )
    differences = queries[:, None, :] - projections
    distances_squared = np.sum(differences * differences, axis=2)
    indices = np.argmin(distances_squared, axis=1)
    rows = np.arange(len(queries))
    selected_ratios = ratios[rows, indices]
    selected_segments = segments[indices]
    selected_lengths = np.sqrt(lengths_squared[indices])
    tangents = np.divide(
        selected_segments,
        selected_lengths[:, None],
        out=np.tile(np.asarray([[1.0, 0.0]]), (len(queries), 1)),
        where=selected_lengths[:, None] > 1e-12,
    )
    selected_residuals = differences[rows, indices]
    arc = _arc_from_clean(points)
    return {
        "s": arc[indices] + selected_ratios * selected_lengths,
        "d": (
            tangents[:, 0] * selected_residuals[:, 1]
            - tangents[:, 1] * selected_residuals[:, 0]
        ),
        "distance": np.sqrt(distances_squared[rows, indices]),
        "point": projections[rows, indices],
        "tangent": tangents,
        "segmentIndex": indices,
    }


def path_curvatures(line, window):
    points = clean_xy(line)
    if len(points) < 2:
        return np.zeros(len(points), dtype=float)
    arc = _arc_from_clean(points)
    total = float(arc[-1])
    half = max(float(window), 0.2) / 2.0
    headings = np.unwrap(np.arctan2(
        np.diff(points[:, 1]), np.diff(points[:, 0])
    ))
    heading_s = (arc[:-1] + arc[1:]) / 2.0
    lower = np.maximum(0.0, arc - half)
    upper = np.minimum(total, arc + half)
    if total > 2.0 * half:
        near_start = arc < half
        near_end = arc > total - half
        upper[near_start] = 2.0 * half
        lower[near_end] = total - 2.0 * half
    # Interpolate segment headings at the exact arc-length window boundaries.
    # This makes their difference consistent with the distance denominator.
    lower_heading = np.interp(lower, heading_s, headings)
    upper_heading = np.interp(upper, heading_s, headings)
    delta = upper_heading - lower_heading
    return np.abs(delta) / np.maximum(upper - lower, 1e-9)


def frenet_hermite_connector(
        start, start_heading, start_reference_s,
        end, end_heading, end_reference_s,
        reference, spacing):
    """Build a cubic Hermite connector in the reference Frenet frame."""
    ref = clean_xy(reference)
    first = np.asarray(start, dtype=float)[:2]
    last = np.asarray(end, dtype=float)[:2]
    start_s = float(start_reference_s)
    end_s = float(end_reference_s)
    span = end_s - start_s
    if span <= 1e-9:
        return clean_xy(np.vstack((first, last)))

    reference_total = float(arc_lengths(ref)[-1])

    def reference_state(point, heading, value):
        base = point_at_s(ref, value)
        tangent = tangent_at_s(ref, value)
        reference_heading = math.atan2(tangent[1], tangent[0])
        normal = np.asarray([-tangent[1], tangent[0]])
        lateral = float(np.dot(point - base, normal))

        curvature_step = max(0.1, min(float(spacing), 0.5))
        lower = max(0.0, value - curvature_step)
        upper = min(reference_total, value + curvature_step)
        if upper <= lower + 1e-9:
            reference_curvature = 0.0
        else:
            lower_heading = heading_at_s(ref, lower)
            upper_heading = heading_at_s(ref, upper)
            heading_delta = math.atan2(
                math.sin(upper_heading - lower_heading),
                math.cos(upper_heading - lower_heading),
            )
            reference_curvature = heading_delta / (upper - lower)

        relative_heading = math.atan2(
            math.sin(float(heading) - reference_heading),
            math.cos(float(heading) - reference_heading),
        )
        lateral_slope = (
            (1.0 - reference_curvature * lateral)
            * math.tan(relative_heading)
        )
        return lateral, lateral_slope

    start_d, start_slope = reference_state(
        first, start_heading, start_s)
    end_d, end_slope = reference_state(last, end_heading, end_s)
    count = max(
        3, int(math.ceil(span / max(float(spacing), 0.1))) + 1)
    sample_s = np.linspace(start_s, end_s, count)
    parameter = (sample_s - start_s) / span
    h00 = 2.0 * parameter ** 3 - 3.0 * parameter ** 2 + 1.0
    h10 = parameter ** 3 - 2.0 * parameter ** 2 + parameter
    h01 = -2.0 * parameter ** 3 + 3.0 * parameter ** 2
    h11 = parameter ** 3 - parameter ** 2
    lateral = (
        h00 * start_d
        + h10 * span * start_slope
        + h01 * end_d
        + h11 * span * end_slope
    )
    output = []
    for value, offset in zip(sample_s, lateral):
        base = point_at_s(ref, value)
        tangent = tangent_at_s(ref, value)
        normal = np.asarray([-tangent[1], tangent[0]])
        output.append(base + offset * normal)
    result = np.asarray(output, dtype=float)
    result[0] = first
    result[-1] = last
    return resample_polyline(result, spacing)


def frenet_connector(current, target, reference, spacing):
    ref = clean_xy(reference)
    start_projection = project_to_polyline(current, ref)
    target_projection = project_to_polyline(target, ref)
    start_s = float(start_projection["s"])
    end_s = max(start_s, float(target_projection["s"]))
    sample_s = np.arange(start_s, end_s, max(float(spacing), 0.1))
    if len(sample_s) == 0 or end_s - sample_s[-1] > 1e-9:
        sample_s = np.append(sample_s, end_s)
    denominator = max(end_s - start_s, 1e-9)
    ratio = np.clip((sample_s - start_s) / denominator, 0.0, 1.0)
    smooth = ratio * ratio * (3.0 - 2.0 * ratio)
    lateral = (
        float(start_projection["d"])
        + smooth * (float(target_projection["d"]) - float(start_projection["d"]))
    )
    output = []
    for value, offset in zip(sample_s, lateral):
        base = point_at_s(ref, value)
        tangent = tangent_at_s(ref, value)
        normal = np.asarray([-tangent[1], tangent[0]])
        output.append(base + offset * normal)
    result = np.asarray(output, dtype=float)
    result[0] = np.asarray(current, dtype=float)[:2]
    result[-1] = np.asarray(target, dtype=float)[:2]
    return resample_polyline(result, spacing)


def _arc_from_clean(points):
    return np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(points, axis=0), axis=1))))


def _points_at_s_clean(points, arc, targets):
    values = np.clip(np.asarray(targets, dtype=float), 0.0, arc[-1])
    indices = np.minimum(
        np.searchsorted(arc, values, side="right") - 1,
        len(points) - 2,
    )
    indices = np.maximum(indices, 0)
    lengths = arc[indices + 1] - arc[indices]
    ratios = np.divide(
        values - arc[indices],
        lengths,
        out=np.zeros(len(values), dtype=float),
        where=lengths > 1e-12,
    )
    return points[indices] + ratios[:, None] * (
        points[indices + 1] - points[indices]
    )
