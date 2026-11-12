"""World-frame IMU loading, stationarity and velocity integration from src/export_route.py."""
import csv

import numpy as np


CSV_COLUMNS = ('time', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z')


def load_input(path):
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as z:
            if all(k in z for k in ('time', 'gyro_world', 'acc_world')):
                time = np.asarray(z['time'], float)
                feat = np.concatenate([z['gyro_world'], z['acc_world']], axis=1).astype(float)
            else:
                time, feat = np.asarray(z['tm'], float), np.asarray(z['feat'], float)
    elif path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or not set(CSV_COLUMNS).issubset(reader.fieldnames):
                raise ValueError('Required CSV columns: ' + ','.join(CSV_COLUMNS))
            values = np.asarray([[float(row[k]) for k in CSV_COLUMNS] for row in reader], float)
        if values.ndim != 2:
            raise ValueError('The CSV contains no valid samples.')
        time, feat = values[:, 0], values[:, 1:]
    else:
        raise ValueError('Input must be a world-frame .npz or .csv file.')
    if time.ndim != 1 or feat.shape != (len(time), 6) or len(time) < 211:
        raise ValueError('At least 211 samples are required. Expected time [N] and IMU channels [N, 6].')
    if not np.isfinite(time).all() or not np.isfinite(feat).all():
        raise ValueError('Timestamps and IMU channels must contain finite values.')
    dt = np.diff(time)
    if np.any(dt <= 0) or not np.allclose(dt, .005, rtol=.02, atol=1e-6):
        raise ValueError('Synchronize and resample to 200 Hz. Timestamps must increase strictly and use seconds.')
    return time, feat


def stationary_intervals(time, feat, window=.5, min_duration=1., gyro_threshold=.05,
                         acc_threshold=.35, gravity=9.81, exit_factor=1.5):
    """Detect stationary intervals using centered RMS windows and hysteresis.

    Labels are -1 for incomplete boundary windows, 0 for nonstationary samples,
    and 1 for stationary samples. The thresholds describe device motion.
    """
    settings = [window, min_duration, gyro_threshold, acc_threshold, gravity, exit_factor]
    if not np.isfinite(settings).all() or min(settings) <= 0 or exit_factor < 1:
        raise ValueError('Window duration, minimum duration, thresholds, and gravity must be positive. The exit factor must be at least 1.')
    dt = float(np.median(np.diff(time)))
    width = max(3, int(round(window / dt)) | 1)
    labels = np.full(len(time), -1, dtype=int)
    if width > len(time):
        raise ValueError('The stationarity window is longer than the recording.')
    residual = feat[:, 3:] - np.array([0., 0., gravity])
    energy = np.column_stack((np.sum(feat[:, :3] ** 2, axis=1),
                              np.sum(residual ** 2, axis=1)))
    cumulative = np.vstack((np.zeros((1, 2)), np.cumsum(energy, axis=0)))
    rms = np.sqrt(np.maximum((cumulative[width:] - cumulative[:-width]) / width, 0))
    half = width // 2
    active = False
    thresholds = np.array([gyro_threshold, acc_threshold])
    for i, values in enumerate(rms):
        active = bool(np.all(values <= thresholds * (exit_factor if active else 1.)))
        labels[i + half] = int(active)
    edges = np.diff(np.r_[False, labels == 1, False].astype(int))
    intervals = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        end = min(float(time[stop - 1] + dt), float(time[-1]))
        if end - time[start] + 1e-9 < min_duration:
            labels[start:stop] = 0
        else:
            intervals.append((float(time[start]), end))
    return labels, intervals


def integrate(time, ids, velocity):
    # Integrate the predicted velocities directly at their original timestamps.
    tp = time[ids]
    dt = np.diff(tp, prepend=tp[0] - np.median(np.diff(tp)))
    return tp, np.cumsum(velocity * dt[:, None], axis=0)
