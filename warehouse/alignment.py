"""Rigid display alignment and the historical handheld endpoint diagnostic."""
import numpy as np


def align_start_yaw(est, gt):
    """Match the start points and fit the optimal planar rotation about the start.

    The rotation is atan2(b, a), where a is the sum of dot products and b is
    the sum of signed cross products of the centered point pairs."""
    e = np.asarray(est, float); g = np.asarray(gt, float)
    n = min(len(e), len(g))
    p = e[:n] - e[0]; q = g[:n] - g[0]
    a = float((p[:, 0] * q[:, 0] + p[:, 1] * q[:, 1]).sum())
    b = float((p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]).sum())
    th = np.arctan2(b, a) if (a * a + b * b) > 1e-20 else 0.0
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return ((np.asarray(est, float) - e[0]) @ R.T) + g[0], float(np.degrees(th))


def initial_heading(xy, distance=3.):
    """Direction of the first 3 m of predicted travel, without using ground truth."""
    xy = np.asarray(xy, float) - xy[0]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if not np.isfinite(xy).all() or arc[-1] < .5:
        raise ValueError('Insufficient displacement to determine initial heading')
    keep = np.r_[True, np.diff(arc) > 0]
    arc, xy = arc[keep], xy[keep]
    assert np.all(np.diff(arc) > 0)
    at = min(distance, arc[-1])
    vector = np.array([np.interp(at, arc, xy[:, j]) for j in range(2)])
    if np.linalg.norm(vector) < .25:
        raise ValueError('Ambiguous initial heading (initial motion doubles back)')
    return float(np.arctan2(vector[1], vector[0]))


def rotate(xy, angle):
    c, s = np.cos(angle), np.sin(angle)
    return (xy - xy[0]) @ np.array([[c, s], [-s, c]])


def display_align(xy, reference):
    """Match start and direction of the first 3 m by a single rigid transform."""
    target = initial_heading(reference)
    angle = float(np.angle(np.exp(1j * (target - initial_heading(xy)))))
    return rotate(xy, angle) + reference[0]


def handheld_endpoint(xy, dense):
    """Replicate the manuscript’s retrospective whole-track yaw fit."""
    count = min(len(xy), 4000)
    indices = np.linspace(0, len(xy) - 1, count).astype(int)
    u = np.linspace(0, 1, count)
    original_u = np.linspace(0, 1, len(dense))
    paired = np.column_stack([np.interp(u, original_u, dense[:, axis]) for axis in (0, 1)])
    aligned, angle = align_start_yaw(xy[indices], paired)
    return rotate(xy, np.deg2rad(angle)) + dense[0], float(np.linalg.norm(aligned[-1] - paired[-1]))
