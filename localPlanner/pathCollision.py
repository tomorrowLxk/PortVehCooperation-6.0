"""polygon在格点建图、正常path验证和Frenet截停中的三种独立处理。"""

from dataclasses import dataclass
import math

import numpy as np

from localPlanner.pathGeometry import (
    arc_lengths,
    clean_xy,
    point_at_s,
    resample_polyline,
)


@dataclass(frozen=True)
class BlockedInterval:
    startS: float
    endS: float
    obstacleIndex: int


def polygon_xy(value):
    polygon = value.get("polygon", value) if isinstance(value, dict) else value
    points = []
    for point in polygon or []:
        if isinstance(point, dict):
            if "x" in point and "y" in point:
                points.append((float(point["x"]), float(point["y"])))
            elif "longitude" in point and "latitude" in point:
                points.append((
                    float(point["longitude"]), float(point["latitude"])
                ))
        else:
            raw = np.asarray(point, dtype=float).reshape(-1)
            if raw.size >= 2:
                points.append((float(raw[0]), float(raw[1])))
    return clean_xy(np.asarray(points, dtype=float))


def point_in_polygon(point, polygon):
    query = np.asarray(point, dtype=float)[:2]
    fence = clean_xy(polygon)
    return _point_in_polygon_clean(query, fence)


def _point_in_polygon_clean(query, fence):
    return bool(_points_in_polygon_clean(
        np.asarray(query, dtype=float).reshape(1, 2), fence)[0])


def _points_in_polygon_clean(queries, fence):
    """保持原射线法和边界判据不变，一次判断多个点。"""
    points = np.asarray(queries, dtype=float).reshape(-1, 2)
    starts = fence
    ends = np.roll(fence, -1, axis=0)
    segments = ends - starts
    lengths_squared = np.sum(segments * segments, axis=1)
    residuals = points[:, None, :] - starts[None, :, :]
    ratios = np.divide(
        np.sum(residuals * segments[None, :, :], axis=2),
        lengths_squared[None, :],
        out=np.zeros((len(points), len(fence)), dtype=float),
        where=lengths_squared[None, :] > 1e-15,
    )
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = starts[None, :, :] + ratios[:, :, None] * segments
    on_boundary = np.any(
        np.sum((points[:, None, :] - projections) ** 2, axis=2)
        <= 1e-18,
        axis=1,
    )

    query_x = points[:, 0, None]
    query_y = points[:, 1, None]
    crosses_y = ((starts[None, :, 1] > query_y)
                 != (ends[None, :, 1] > query_y))
    denominator = ends[:, 1] - starts[:, 1]
    crossing_x = np.divide(
        (ends[:, 0] - starts[:, 0])[None, :]
        * (query_y - starts[None, :, 1]),
        denominator[None, :],
        out=np.zeros((len(points), len(fence)), dtype=float),
        where=np.abs(denominator[None, :]) > 0.0,
    ) + starts[None, :, 0]
    inside = np.count_nonzero(
        crosses_y & (query_x < crossing_x), axis=1) % 2 == 1
    return on_boundary | inside


def line_blocked_intervals(
        line, polygons, vehicle_length, vehicle_width,
        longitudinal_uncertainty, lateral_uncertainty, lateral_margin,
        stop_distance, rear_clearance, sample_spacing):
    """按中心线点是否落入polygon生成格点图禁止布点、禁止跨越区间。"""
    if not polygons:
        return []
    dense = resample_polyline(line, sample_spacing)
    arc = arc_lengths(dense)
    half_length = (
        0.5 * float(vehicle_length) + float(longitudinal_uncertainty)
    )
    half_width = (
        0.5 * float(vehicle_width)
        + float(lateral_uncertainty)
        + float(lateral_margin)
    )
    rectangles = _vehicle_rectangles(dense, half_length, half_width)
    rectangle_min = np.min(rectangles, axis=1)
    rectangle_max = np.max(rectangles, axis=1)
    raw = []
    fences = [clean_xy(polygon) for polygon in polygons]
    for obstacle_index, fence in enumerate(fences):
        mask = _vehicle_occupancy_mask(
            fence, rectangles, rectangle_min, rectangle_max)
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        group_start = int(indices[0])
        previous = group_start
        for raw_index in indices[1:]:
            index = int(raw_index)
            if index != previous + 1:
                raw.append((
                    float(arc[group_start]), float(arc[previous]),
                    obstacle_index,
                ))
                group_start = index
            previous = index
        raw.append((
            float(arc[group_start]), float(arc[previous]), obstacle_index
        ))

    upstream = float(stop_distance)
    downstream = float(rear_clearance)
    total = float(arc[-1])
    expanded = [
        BlockedInterval(
            max(0.0, start - upstream),
            min(total, end + downstream),
            obstacle_index,
        )
        for start, end, obstacle_index in raw
    ]
    return _merge_intervals(expanded)


def _vehicle_occupancy_mask(
        fence, rectangles, rectangle_min, rectangle_max):
    # Batch collision mask for vehicle rectangles and their swept hulls.
    fence_min = np.min(fence, axis=0)
    fence_max = np.max(fence, axis=0)
    mask = np.zeros(len(rectangles), dtype=bool)
    candidates = np.flatnonzero(np.all(
        (rectangle_max >= fence_min) & (rectangle_min <= fence_max),
        axis=1,
    ))
    if len(candidates):
        mask[candidates] = _polygons_intersect_many(
            rectangles[candidates], fence)

    if len(rectangles) < 2:
        return mask
    sweep_min = np.minimum(rectangle_min[:-1], rectangle_min[1:])
    sweep_max = np.maximum(rectangle_max[:-1], rectangle_max[1:])
    sweep_candidates = np.flatnonzero(np.all(
        (sweep_max >= fence_min) & (sweep_min <= fence_max), axis=1,
    ))
    sweep_groups = {}
    for index in sweep_candidates:
        sweep = _convex_hull(np.vstack(
            (rectangles[index], rectangles[index + 1])))
        sweep_groups.setdefault(len(sweep), []).append((int(index), sweep))
    for sweeps in sweep_groups.values():
        hits = _polygons_intersect_many(
            np.asarray([sweep for _, sweep in sweeps], dtype=float), fence)
        for (index, _), hit in zip(sweeps, hits):
            if hit:
                mask[index:index + 2] = True
    return mask


def interval_contains(intervals, value):
    return any(
        item.startS - 1e-9 <= value <= item.endS + 1e-9
        for item in intervals
    )


def interval_crossed(intervals, start_s, end_s):
    lower, upper = sorted((float(start_s), float(end_s)))
    return any(
        lower < item.endS - 1e-9 and upper > item.startS + 1e-9
        for item in intervals
    )


def path_vehicle_intersects(
        path, polygons, vehicle_length, vehicle_width,
        longitudinal_uncertainty, lateral_uncertainty,
        lateral_margin, sample_spacing):
    """用本车扩张矩形沿稠密曲线的扫掠区域验证正常path。"""
    if not polygons:
        return False
    raw_path = clean_xy(path)
    if len(raw_path) == 0:
        return False
    half_length = 0.5 * float(vehicle_length) + float(
        longitudinal_uncertainty)
    half_width = (
            0.5 * float(vehicle_width)
            + float(lateral_uncertainty)
            + float(lateral_margin)
    )
    sweep_radius = math.hypot(half_length, half_width)
    raw_min = np.min(raw_path, axis=0) - sweep_radius
    raw_max = np.max(raw_path, axis=0) + sweep_radius
    fences = []
    for polygon in polygons:
        fence = clean_xy(polygon)
        fence_min = np.min(fence, axis=0)
        fence_max = np.max(fence, axis=0)
        if np.any(fence_max < raw_min) or np.any(fence_min > raw_max):
            continue
        fences.append((fence, fence_min, fence_max))
    if not fences:
        return False

    points = resample_polyline(raw_path, sample_spacing)
    path_min = np.min(points, axis=0) - sweep_radius
    path_max = np.max(points, axis=0) + sweep_radius
    rectangles = _vehicle_rectangles(points, half_length, half_width)
    rectangle_min = np.min(rectangles, axis=1)
    rectangle_max = np.max(rectangles, axis=1)
    for fence, fence_min, fence_max in fences:
        if np.any(fence_max < path_min) or np.any(fence_min > path_max):
            continue
        rectangle_candidates = np.flatnonzero(np.all(
            (rectangle_max >= fence_min) & (rectangle_min <= fence_max),
            axis=1,
        ))
        if len(rectangle_candidates) and np.any(
                _polygons_intersect_many(
                    rectangles[rectangle_candidates], fence)):
            return True
        sweep_min = np.minimum(rectangle_min[:-1], rectangle_min[1:])
        sweep_max = np.maximum(rectangle_max[:-1], rectangle_max[1:])
        sweep_candidates = np.flatnonzero(np.all(
            (sweep_max >= fence_min) & (sweep_min <= fence_max), axis=1,
        ))
        sweep_groups = {}
        for index in sweep_candidates:
            sweep = _convex_hull(np.vstack(
                (rectangles[index], rectangles[index + 1])))
            sweep_groups.setdefault(len(sweep), []).append(sweep)
        for sweeps in sweep_groups.values():
            if np.any(_polygons_intersect_many(
                    np.asarray(sweeps, dtype=float), fence)):
                return True
    return False


def first_polygon_stop_s(
        path, polygons, vehicle_length, longitudinal_uncertainty,
        stop_distance, sample_spacing):
    """Frenet降级path只在线上新增T并截停，不做车辆形状碰撞。"""
    if not polygons:
        return None
    dense = resample_polyline(path, sample_spacing)
    arc = arc_lengths(dense)
    fences = [clean_xy(polygon) for polygon in polygons]
    occupied = np.zeros(len(dense), dtype=bool)
    for fence in fences:
        occupied |= _points_in_polygon_clean(dense, fence)
    for index in np.flatnonzero(occupied):
        return max(
            0.0,
            float(arc[index])
            - 0.5 * float(vehicle_length)
            - float(longitudinal_uncertainty)
            - float(stop_distance),
        )
    return None


def vehicle_polygon(position, heading, length, width,
                    longitudinal_uncertainty, lateral_uncertainty):
    center = np.asarray(position, dtype=float)[:2]
    tangent = np.asarray([math.cos(float(heading)), math.sin(float(heading))])
    normal = np.asarray([-tangent[1], tangent[0]])
    half_length = 0.5 * float(length) + float(longitudinal_uncertainty)
    half_width = 0.5 * float(width) + float(lateral_uncertainty)
    return np.asarray([
        center + half_length * tangent + half_width * normal,
        center + half_length * tangent - half_width * normal,
        center - half_length * tangent - half_width * normal,
        center - half_length * tangent + half_width * normal,
    ])


def polygons_intersect(first, second):
    left = clean_xy(first)
    right = clean_xy(second)
    return _polygons_intersect_clean(left, right)


def _polygons_intersect_clean(left, right):
    if len(left) < 3 or len(right) < 3:
        return False
    if (
            np.max(left[:, 0]) < np.min(right[:, 0])
            or np.max(right[:, 0]) < np.min(left[:, 0])
            or np.max(left[:, 1]) < np.min(right[:, 1])
            or np.max(right[:, 1]) < np.min(left[:, 1])
    ):
        return False
    if np.any(_points_in_polygon_clean(left, right)):
        return True
    if np.any(_points_in_polygon_clean(right, left)):
        return True
    return _polygon_edges_intersect(left, right)


def _polygons_intersect_many(lefts, right):
    """批量判断若干同顶点数多边形与同一个polygon是否相交。"""
    polygons = np.asarray(lefts, dtype=float)
    fence = np.asarray(right, dtype=float)
    count = len(polygons)
    result = np.zeros(count, dtype=bool)
    polygon_min = np.min(polygons, axis=1)
    polygon_max = np.max(polygons, axis=1)
    fence_min = np.min(fence, axis=0)
    fence_max = np.max(fence, axis=0)
    active = np.flatnonzero(np.all(
        (polygon_max >= fence_min) & (polygon_min <= fence_max), axis=1))
    if not len(active):
        return result
    candidates = polygons[active]

    left_inside = _points_in_polygon_clean(
        candidates.reshape(-1, 2), fence
    ).reshape(len(candidates), candidates.shape[1]).any(axis=1)
    right_inside = _common_points_in_polygons(fence, candidates).any(axis=1)
    edge_cross = _polygon_edges_intersect_many(candidates, fence)
    result[active] = left_inside | right_inside | edge_cross
    return result


def _common_points_in_polygons(points, polygons):
    """判断同一组点分别位于每一个polygon内的情况。"""
    queries = np.asarray(points, dtype=float)
    starts = polygons
    ends = np.roll(polygons, -1, axis=1)
    segments = ends - starts
    lengths_squared = np.sum(segments * segments, axis=2)
    residuals = queries[None, :, None, :] - starts[:, None, :, :]
    ratios = np.divide(
        np.sum(residuals * segments[:, None, :, :], axis=3),
        lengths_squared[:, None, :],
        out=np.zeros((len(polygons), len(queries), polygons.shape[1])),
        where=lengths_squared[:, None, :] > 1e-15,
    )
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = (
            starts[:, None, :, :]
            + ratios[:, :, :, None] * segments[:, None, :, :]
    )
    on_boundary = np.any(np.sum(
        (queries[None, :, None, :] - projections) ** 2, axis=3
    ) <= 1e-18, axis=2)

    query_x = queries[None, :, None, 0]
    query_y = queries[None, :, None, 1]
    crosses_y = ((starts[:, None, :, 1] > query_y)
                 != (ends[:, None, :, 1] > query_y))
    denominator = ends[:, :, 1] - starts[:, :, 1]
    crossing_x = np.divide(
        (ends[:, :, 0] - starts[:, :, 0])[:, None, :]
        * (query_y - starts[:, None, :, 1]),
        denominator[:, None, :],
        out=np.zeros((len(polygons), len(queries), polygons.shape[1])),
        where=np.abs(denominator[:, None, :]) > 0.0,
    ) + starts[:, None, :, 0]
    inside = np.count_nonzero(
        crosses_y & (query_x < crossing_x), axis=2) % 2 == 1
    return on_boundary | inside


def _polygon_edges_intersect_many(lefts, right):
    a = lefts
    b = np.roll(lefts, -1, axis=1)
    c = right
    d = np.roll(right, -1, axis=0)
    ab = b - a
    cd = d - c

    def cross(first, second):
        return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]

    first = cross(
        ab[:, :, None, :], c[None, None, :, :] - a[:, :, None, :])
    second = cross(
        ab[:, :, None, :], d[None, None, :, :] - a[:, :, None, :])
    third = cross(
        cd[None, None, :, :], a[:, :, None, :] - c[None, None, :, :])
    fourth = cross(
        cd[None, None, :, :], b[:, :, None, :] - c[None, None, :, :])
    proper = np.any(
        (first * second < -1e-12) & (third * fourth < -1e-12),
        axis=(1, 2),
    )

    def within(points, starts, ends):
        return np.all(
            (points >= np.minimum(starts, ends) - 1e-12)
            & (points <= np.maximum(starts, ends) + 1e-12),
            axis=3,
        )

    a_grid = a[:, :, None, :]
    b_grid = b[:, :, None, :]
    c_grid = c[None, None, :, :]
    d_grid = d[None, None, :, :]
    touching = np.any(
        (np.abs(first) <= 1e-12) & within(c_grid, a_grid, b_grid)
        | (np.abs(second) <= 1e-12) & within(d_grid, a_grid, b_grid)
        | (np.abs(third) <= 1e-12) & within(a_grid, c_grid, d_grid)
        | (np.abs(fourth) <= 1e-12) & within(b_grid, c_grid, d_grid),
        axis=(1, 2),
    )
    return proper | touching


def _polygon_edges_intersect(left, right):
    a = left
    b = np.roll(left, -1, axis=0)
    c = right
    d = np.roll(right, -1, axis=0)
    ab = b - a
    cd = d - c

    def cross(first, second):
        return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]

    first = cross(ab[:, None, :], c[None, :, :] - a[:, None, :])
    second = cross(ab[:, None, :], d[None, :, :] - a[:, None, :])
    third = cross(cd[None, :, :], a[:, None, :] - c[None, :, :])
    fourth = cross(cd[None, :, :], b[:, None, :] - c[None, :, :])
    if np.any((first * second < -1e-12) & (third * fourth < -1e-12)):
        return True

    def within(points, starts, ends):
        return np.all(
            (points >= np.minimum(starts, ends) - 1e-12)
            & (points <= np.maximum(starts, ends) + 1e-12),
            axis=2,
        )

    a_grid = a[:, None, :]
    b_grid = b[:, None, :]
    c_grid = c[None, :, :]
    d_grid = d[None, :, :]
    return bool(np.any(
        (np.abs(first) <= 1e-12) & within(c_grid, a_grid, b_grid)
        | (np.abs(second) <= 1e-12) & within(d_grid, a_grid, b_grid)
        | (np.abs(third) <= 1e-12) & within(a_grid, c_grid, d_grid)
        | (np.abs(fourth) <= 1e-12) & within(b_grid, c_grid, d_grid)
    ))


def _vehicle_rectangles(points, half_length, half_width):
    if len(points) == 1:
        tangents = np.asarray([[1.0, 0.0]])
    else:
        tangents = np.empty_like(points)
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        if len(points) > 2:
            tangents[1:-1] = points[2:] - points[:-2]
        norms = np.linalg.norm(tangents, axis=1)
        tangents = np.divide(
            tangents, norms[:, None],
            out=np.tile(np.asarray([[1.0, 0.0]]), (len(points), 1)),
            where=norms[:, None] > 1e-12,
        )
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    forward = float(half_length) * tangents
    lateral = float(half_width) * normals
    return np.stack((
        points + forward + lateral,
        points + forward - lateral,
        points - forward - lateral,
        points - forward + lateral,
    ), axis=1)


def _merge_intervals(intervals):
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: (item.startS, item.endS))
    output = [ordered[0]]
    for item in ordered[1:]:
        previous = output[-1]
        if item.startS <= previous.endS + 1e-9:
            output[-1] = BlockedInterval(
                previous.startS,
                max(previous.endS, item.endS),
                previous.obstacleIndex,
            )
        else:
            output.append(item)
    return output


def _convex_hull(points):
    unique = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float)

    def cross(origin, first, second):
        return (
                (first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)
