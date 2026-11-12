"""Paths, fixed paper inputs and shared output helpers."""
import csv
import hashlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper_reference'
ROUTES = ('Path1', 'Path2', 'Path3')
PAPER_CARRYING_MODES = ('Hand_Swing', 'Trouser_Pocket', 'Jacket_Pocket', 'Freedom')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fields=None):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields if fields is not None else list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def check_sha256(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'SHA-256 checksum mismatch: {path}')


@contextmanager
def new_output(path):
    if path.exists():
        raise FileExistsError(f'Output directory already exists. Choose a new path: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    working = Path(mkdtemp(prefix=f'.{path.name}-', dir=path.parent))
    try:
        yield working
        if path.exists():
            raise FileExistsError(path)
        working.rename(path)
    finally:
        if working.exists():
            shutil.rmtree(working)


def reference(route):
    if route not in ROUTES:
        raise ValueError(f'Unknown reference route: {route}')
    folder = ROOT / 'data/reference'
    rows = read_csv(folder / f'{route}_reference_polyline.csv')
    xy = np.array([[float(row['x_m']), float(row['y_m'])] for row in rows])
    length = sum(float(row['segment_length_m']) for row in rows)
    if length <= 0 or not np.isfinite(length):
        raise ValueError(f'{route}: Invalid reference path length.')
    return xy, length


def track_length(xy):
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError('A trajectory must contain at least two finite two-dimensional coordinates.')
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
