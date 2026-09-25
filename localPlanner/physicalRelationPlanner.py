"""根据车辆当前位置和最终path提取不可反转的物理前后关系。"""

import math

import numpy as np

from localPlanner.pathGeometry import clean_xy
from parameters import (
    PHYSICAL_FRONT_REAR_HEADING_THRESHOLD_DEG,
    PHYSICAL_FRONT_REAR_MIN_PROGRESS,
)


def determine_physical_front_rear_edges(
        cooperation_blocks,
        selected_paths,
        vehicle_static_info,
        vehicle_dynamic_info):
    """返回(前车ID, 后车ID)物理硬约束边。"""
    edges = []
    path_data = {
        vehicle_id: _prepare_path(path)
        for vehicle_id, path in selected_paths.items()
    }

    for block in cooperation_blocks:
        members = list(block)
        progress = _block_ahead_progress(
            members, path_data, vehicle_static_info, vehicle_dynamic_info)

        for first_index, first_id in enumerate(members):
            for second_index in range(first_index + 1, len(members)):
                second_id = members[second_index]
                first_ahead = progress[first_index, second_index]
                second_ahead = progress[second_index, first_index]

                if np.isfinite(first_ahead) and not np.isfinite(second_ahead):
                    edges.append((first_id, second_id))
                elif np.isfinite(second_ahead) and not np.isfinite(first_ahead):
                    edges.append((second_id, first_id))
                elif first_ahead > (
                        second_ahead + PHYSICAL_FRONT_REAR_MIN_PROGRESS):
                    edges.append((first_id, second_id))
                elif second_ahead > (
                        first_ahead + PHYSICAL_FRONT_REAR_MIN_PROGRESS):
                    edges.append((second_id, first_id))
    return edges


def _block_ahead_progress(
        members,
        path_data,
        vehicle_static_info,
        vehicle_dynamic_info):
    count = len(members)
    progress = np.full((count, count), np.nan, dtype=float)
    positions = np.asarray([
        vehicle_dynamic_info[vehicle_id]["position"][:2]
        for vehicle_id in members
    ], dtype=float)
    headings = np.asarray([
        vehicle_dynamic_info[vehicle_id]["heading"]
        for vehicle_id in members
    ], dtype=float)
    half_widths = np.asarray([
        0.5 * float(vehicle_static_info[vehicle_id]["width"])
        for vehicle_id in members
    ])
    lateral_uncertainties = np.asarray([
        vehicle_dynamic_info[vehicle_id]["lateralUncertainty"]
        for vehicle_id in members
    ], dtype=float)

    for rear_index, rear_id in enumerate(members):
        rear_path = path_data[rear_id]
        if len(rear_path["points"]) < 2:
            continue

        lateral_thresholds = (
            half_widths
            + half_widths[rear_index]
            + lateral_uncertainties
            + lateral_uncertainties[rear_index]
        )
        path_lower = np.min(rear_path["points"], axis=0)
        path_upper = np.max(rear_path["points"], axis=0)
        aabb_margin = (
            lateral_thresholds + PHYSICAL_FRONT_REAR_MIN_PROGRESS)
        candidates = np.flatnonzero(np.all(
            (positions >= path_lower - aabb_margin[:, None])
            & (positions <= path_upper + aabb_margin[:, None]),
            axis=1,
        ))
        candidates = candidates[candidates != rear_index]
        if len(candidates) == 0:
            continue

        projection = _project_many_to_path(
            positions[candidates], rear_path)
        residuals = positions[candidates] - projection["points"]
        longitudinal_residuals = np.abs(np.sum(
            residuals * projection["tangents"], axis=1))
        path_headings = np.arctan2(
            projection["tangents"][:, 1],
            projection["tangents"][:, 0],
        )
        heading_errors = np.abs(np.arctan2(
            np.sin(headings[candidates] - path_headings),
            np.cos(headings[candidates] - path_headings),
        ))
        valid = (
            (projection["s"] > PHYSICAL_FRONT_REAR_MIN_PROGRESS)
            & (longitudinal_residuals <= PHYSICAL_FRONT_REAR_MIN_PROGRESS)
            & (projection["distances"]
               <= lateral_thresholds[candidates])
            & (heading_errors <= math.radians(
                PHYSICAL_FRONT_REAR_HEADING_THRESHOLD_DEG))
        )
        progress[candidates[valid], rear_index] = projection["s"][valid]
    return progress


def _prepare_path(path):
    points = clean_xy(path)
    if len(points) < 2:
        return {"points": points}

    segments = np.diff(points, axis=0)
    lengths_squared = np.sum(segments * segments, axis=1)
    segment_lengths = np.sqrt(lengths_squared)
    return {
        "points": points,
        "segments": segments,
        "lengthsSquared": lengths_squared,
        "segmentLengths": segment_lengths,
        "arc": np.r_[0.0, np.cumsum(segment_lengths)],
    }


def _project_many_to_path(query_points, path):
    points = path["points"]
    segments = path["segments"]
    lengths_squared = path["lengthsSquared"]
    residuals = query_points[:, None, :] - points[:-1][None, :, :]
    ratios = np.divide(
        np.sum(residuals * segments[None, :, :], axis=2),
        lengths_squared[None, :],
        out=np.zeros((len(query_points), len(segments)), dtype=float),
        where=lengths_squared[None, :] > 1e-12,
    )
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = (
        points[:-1][None, :, :]
        + ratios[:, :, None] * segments[None, :, :]
    )
    differences = query_points[:, None, :] - projections
    distances_squared = np.sum(differences * differences, axis=2)
    indices = np.argmin(distances_squared, axis=1)
    rows = np.arange(len(query_points))
    selected_lengths = path["segmentLengths"][indices]
    return {
        "s": (
            path["arc"][indices]
            + ratios[rows, indices] * selected_lengths
        ),
        "distances": np.sqrt(distances_squared[rows, indices]),
        "points": projections[rows, indices],
        "tangents": segments[indices] / selected_lengths[:, None],
    }