"""Reconstruct trajectories with the frozen HN-Transformer checkpoint and optionally run the original PDR baseline."""
import argparse
import os
from pathlib import Path

import numpy as np

from warehouse import alignment, hn_model, imu
from warehouse.protocol import (
    ROOT, ROUTES, check_sha256, new_output, read_json, reference, sha256, track_length, write_json,
)


def select_records(args):
    if args.input:
        if not args.route or args.raw_root:
            raise ValueError('Specify --route for a custom input. The PDR baseline is available for the fixed carrying-mode cohort.')
        path = args.input.resolve()
        return [dict(sequence=path.stem.removesuffix('__synced'), cohort='custom', route=args.route,
                     mode='Custom', input=str(path), sha256=sha256(path))]
    if args.route:
        raise ValueError('Use --route with --input.')
    records = read_json(ROOT / 'config/records.json')
    if args.sequence:
        records = [row for row in records if row['sequence'] == args.sequence]
        if not records:
            raise ValueError(f'Sequence {args.sequence} is absent from the study manifest.')
    if args.raw_root and not any(row['cohort'] == 'carrying' for row in records):
        raise ValueError('--raw-root requires at least one recording from the carrying-mode cohort.')
    return records


def add_pdr(cache, args, record, input_path, ref, config):
    from warehouse.pdr import run_pdr, validate_pair

    directory = args.raw_root / record['route'] / record['sequence']
    for filename, digest in record['raw_sha256'].items():
        check_sha256(directory / filename, digest)
    validate_pair(directory, input_path)
    result = run_pdr(directory, seed=config['pdr_seed'], settings=config['pdr'])
    xy = result['xy']
    np.testing.assert_allclose(track_length(xy), result['length_m'], rtol=0, atol=1e-8)
    cache.update(pdr_xy=xy, pdr_time_s=result['time'],
                 pdr_display_xy=alignment.display_align(xy, ref))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--sequence', help='Process one recording listed in the study manifest.')
    selection.add_argument('--input', type=Path, help='Path to a single 200 Hz world-frame NPZ or CSV recording.')
    parser.add_argument('--route', choices=ROUTES, help='Reference route for a custom recording.')
    parser.add_argument('--raw-root', type=Path, help='Root of the PDR input files, containing Path1, Path2, and Path3.')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    records = select_records(args)
    config = read_json(ROOT / 'config/inference.json')
    checkpoint = ROOT / config['checkpoint']
    check_sha256(checkpoint, config['checkpoint_sha256'])
    for name, digest in config['reference_sha256'].items():
        check_sha256(ROOT / 'data/reference' / name, digest)
    os.environ.setdefault('PKG_THREADS', '4')
    device = hn_model.pick_device()
    model, window, rate = hn_model.load_checkpoint(checkpoint, device)

    with new_output(args.out) as out:
        for i, record in enumerate(records, 1):
            sequence, route = record['sequence'], record['route']
            path = ROOT / record['input']
            check_sha256(path, record['sha256'])
            time, features = imu.load_input(path)
            chunk = 256 if record['cohort'] == 'carrying' else 2048
            ids, velocity = hn_model.predict(model, features, device, window, rate, chunk=chunk)
            if ids is None or not np.isfinite(velocity).all():
                raise ValueError(f'{sequence}: Too few valid velocity predictions.')
            timestamps, xy = imu.integrate(time, ids, velocity)
            ref, length = reference(route)
            cache = dict(hn_xy=xy, hn_time_s=timestamps, hn_velocity_mps=velocity, hn_indices=ids,
                         reference_xy=ref, reference_length_m=length)
            if record['cohort'] == 'handheld':
                dense = np.loadtxt(ROOT / f'data/reference/{route}_reference_dense.csv',
                                   delimiter=',', skiprows=1)
                aligned, endpoint = alignment.handheld_endpoint(xy, dense)
                cache.update(hn_display_xy=aligned, hn_endpoint_error_m=endpoint)
            elif record['cohort'] == 'carrying':
                cache['hn_display_xy'] = alignment.display_align(xy, ref)
                if args.raw_root:
                    add_pdr(cache, args, record, path, ref, config)
            track_length(xy)
            target = out / f'{sequence}.npz'
            np.savez_compressed(target, **cache)
            record['cache_sha256'] = sha256(target)
            print(f'[{i}/{len(records)}] {sequence}', flush=True)
        write_json(out / 'manifest.json', dict(
            checkpoint_sha256=config['checkpoint_sha256'], device=str(device),
            original_pdr=bool(args.raw_root), records=records))
    print(f'Trajectories written to {args.out.resolve()}')


if __name__ == '__main__':
    main()
