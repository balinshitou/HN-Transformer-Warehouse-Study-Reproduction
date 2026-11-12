"""Original LBS PDR from src/lbs_pdr_core.py, with the paper’s legacy filter rate."""
import csv
from pathlib import Path

import numpy as np
import pywt
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

from .imu import load_input


def read_sensor(path, gyro=False):
    axes = [f'{a} ({"rad/s" if gyro else "m/s^2"})' for a in 'XYZ']
    with Path(path).open(encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        columns = ['Time (s)', *axes]
        if not reader.fieldnames or not set(columns).issubset(reader.fieldnames):
            raise ValueError(f'{path}: Missing required columns: {columns}')
        values = np.asarray([[float(row[k]) for k in columns] for row in reader], float)
    if (values.ndim != 2 or len(values) < 12 or values.shape[1] != 4
            or not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0)):
        raise ValueError(f'{path}: Expected enough finite samples and strictly increasing timestamps.')
    return values[:, 0], values[:, 1:]


def median_interpolation(time, values, window_size=6, slide_size=3):
    median = np.asarray([np.median(values[i:i + window_size], axis=0)
                         for i in range(0, len(values), slide_size)])
    target = np.linspace(time[0], time[-1], len(median))
    # Fit the spline to time[::slide_size] before evaluating it on a uniform grid.
    return target, CubicSpline(time[::slide_size], median, bc_type='clamped')(target)


def kalman(values, settings):
    estimate = settings['initial_value']
    probability = settings['initial_probability'] + settings['process_variance']
    result = []
    for value in values:
        gain = probability / (probability + settings['measurement_variance'])
        estimate = estimate + gain * (value - estimate)
        result.append(estimate)
        probability = (1 - gain) * probability + settings['process_variance']
    return np.asarray(result)


def wavelet_denoise(values):
    coeffs = pywt.wavedec(values, 'db4', level=3)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745
    # A zero threshold leaves the signal unchanged and avoids division by zero in PyWavelets.
    cleaned = coeffs if threshold == 0 else [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(cleaned,
                       'db4')[:len(values)]


def ampd_peaks(signal):
    """Select peaks using the legacy AMPD scale, boundaries, and index offset.

    The Boolean matrix represents the zero entries in the original local
    maxima scalogram. Select the scale with the most zeros, then retain
    positions that are zero at every scale up to the selected scale.
    """
    signal = np.asarray(signal, float)
    n = len(signal)
    if n < 4:
        raise ValueError('AMPD requires at least four samples.')
    index = np.arange(n)
    slope, intercept = np.polyfit(index, signal, 1)
    detrended = signal - (slope * index + intercept)

    def maxima(k):
        row = np.zeros(n, dtype=bool)
        row[k + 1:n - k + 1] = ((detrended[k:n - k] > detrended[:n - 2 * k])
                                  & (detrended[k:n - k] > detrended[2 * k:]))
        return row

    best_count, best_scale = -1, 1
    for k in range(1, (n + 1) // 2):
        count = int(np.count_nonzero(maxima(k)))
        if count > best_count:
            best_count, best_scale = count, k
    peaks = np.ones(n, dtype=bool)
    for k in range(1, best_scale + 1):
        peaks &= maxima(k)
    return detrended, np.flatnonzero(peaks)


def calculate_step_length(acc_data, previous_length, settings):
    return (settings['step_P'] * previous_length
            + (1 - settings['step_P']) * settings['step_K']
            * (np.sum(np.abs(acc_data)) / len(acc_data)) ** (1 / 3))


def stepstride(peaks, acceleration, initial_length, settings):
    """Estimate stride length from all AMPD peaks before filtering counted steps in run_pdr."""
    lengths = []
    previous = initial_length
    for start, end in zip(peaks[:-1], peaks[1:]):
        segment = acceleration[start:end]
        if (len(segment) > 1 and settings['step_amplitude_min'] <= np.ptp(segment)
                <= settings['step_amplitude_max']):
            previous = calculate_step_length(segment, previous, settings)
            lengths.append(previous)
    if not lengths:
        raise ValueError('Mean stride length cannot be estimated without valid intervals between peaks.')
    return float(np.mean(lengths))


def complementary_filter(time, gyro):
    phi, theta, psi = 0., 0., 0.
    angles = np.zeros_like(gyro)
    special = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    for i in range(1, len(gyro)):
        dt = time[i] - time[i - 1]
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        cos_theta, tan_theta = np.cos(theta), np.tan(theta)
        phi_dot = (gyro[i, 0] + sin_phi * tan_theta * gyro[i, 1]
                   + cos_phi * tan_theta * gyro[i, 2])
        theta_dot = cos_phi * gyro[i, 1] - sin_phi * gyro[i, 2]
        psi_dot = sin_phi / cos_theta * gyro[i, 1] + cos_phi / cos_theta * gyro[i, 2]
        phi += phi_dot * dt
        theta += theta_dot * dt
        psi += psi_dot * dt
        if abs(psi - angles[i - 1, 2]) < .004:
            normalized = psi % (2 * np.pi)
            difference = np.abs(normalized - np.asarray(special))
            psi = special[int(np.argmin(np.minimum(difference, 2 * np.pi - difference)))]
        angles[i] = [phi, theta, psi]
    if not np.isfinite(angles).all():
        raise ValueError('Euler-angle integration produced nonfinite values.')
    return angles


def run_pdr(directory, *, seed, settings):
    ta, acc = read_sensor(Path(directory) / 'Linear Accelerometer.csv')
    tg, gyro = read_sensor(Path(directory) / 'Gyroscope.csv', gyro=True)
    ta, acc = median_interpolation(ta, acc, settings['window_size'], settings['slide_size'])
    tg, gyro = median_interpolation(tg, gyro, settings['window_size'], settings['slide_size'])
    actual_fs = float(1 / np.median(np.diff(ta)))
    filter_fs = settings['legacy_fs_hz']
    if not 0 < settings['cutoff_hz'] < filter_fs / 2:
        raise ValueError(f'The cutoff frequency must be below the Nyquist frequency (sampling rate: {filter_fs:.6f} Hz).')
    b, a = butter(settings['filter_order'], settings['cutoff_hz'], btype='low', fs=filter_fs)
    acc_z = kalman(filtfilt(b, a, acc[:, 2]), settings)
    filtered_gyro = np.column_stack([kalman(wavelet_denoise(gyro[:, j]), settings)
                                     for j in range(3)])
    angles = complementary_filter(tg, filtered_gyro)
    detrended, peaks = ampd_peaks(acc_z)
    initial_length = float(np.random.default_rng(seed).uniform(
        settings['initial_step_min_m'], settings['initial_step_max_m']))
    average = stepstride(peaks, acc_z, initial_length, settings)
    counted = peaks[detrended[peaks] > settings['peak_threshold']]
    if not len(counted):
        raise ValueError('No steps met the detection threshold.')
    if ta[counted[0]] < tg[0] or ta[counted[-1]] > tg[-1]:
        raise ValueError('Step peaks fall outside the gyroscope time range.')
    # Interpolate heading at the step timestamps to account for different sensor time axes.
    yaw = np.interp(ta[counted], tg, np.unwrap(angles[:, 2]))
    delta = np.column_stack((-np.sin(yaw), np.cos(yaw))) * average
    xy = np.vstack((np.zeros(2), np.cumsum(delta, axis=0)))
    return {'xy': xy, 'time': np.r_[ta[0], ta[counted]],
            'length_m': float(average * len(counted)),
            'filter_fs_hz': float(filter_fs), 'actual_fs_hz': actual_fs,
            'initial_step_m': initial_length}


def validate_pair(directory, npz_path, tolerance=1e-7):
    """Validate the CSV and NPZ pairing using the full device-frame arrays."""
    ta, acc = read_sensor(directory / 'Accelerometer.csv')
    tg, gyro = read_sensor(directory / 'Gyroscope.csv', gyro=True)
    read_sensor(directory / 'Linear Accelerometer.csv')
    time, _ = load_input(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        stored_acc, stored_gyro, bias = z['acce'], z['gyro'], z['bias']
    if (stored_acc.shape != (len(time), 3) or stored_gyro.shape != (len(time), 3)
            or bias.shape != (3,) or not all(np.isfinite(a).all()
                                            for a in (stored_acc, stored_gyro, bias))):
        raise ValueError(f'{npz_path.name}: Invalid device-frame arrays or gyroscope bias.')
    interpolated_acc = np.column_stack([np.interp(time, ta, acc[:, j]) for j in range(3)])
    interpolated_gyro = np.column_stack([np.interp(time, tg, gyro[:, j]) for j in range(3)]) - bias
    error_acc = float(np.max(np.abs(interpolated_acc - stored_acc)))
    error_gyro = float(np.max(np.abs(interpolated_gyro - stored_gyro)))
    if max(error_acc, error_gyro) > tolerance:
        raise ValueError(f'Sensor data mismatch between {directory.name} and {npz_path.name}: '
                         f'accelerometer difference={error_acc:.6g}, gyroscope difference={error_gyro:.6g}')
    return {'acc_max_abs_difference_mps2': error_acc,
            'gyro_max_abs_difference_radps': error_gyro}
