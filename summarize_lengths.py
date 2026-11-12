"""Compute trajectory-length ratios and relative path-length errors by route and carrying mode."""
import argparse
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import mean
import numpy as np

from warehouse.protocol import (
    PAPER, PAPER_CARRYING_MODES, ROOT, ROUTES, check_sha256, new_output, read_csv, read_json, track_length, write_csv, write_json,
)


def from_tracks(folder):
    manifest = read_json(folder / 'manifest.json')
    for record in manifest['records']:
        path = folder / (record['sequence'] + '.npz')
        check_sha256(path, record['cache_sha256'])
        with np.load(path, allow_pickle=False) as track:
            keys = {k: record[k] for k in ('cohort', 'sequence', 'route', 'mode')}
            for method in ('hn', 'pdr'):
                if f'{method}_xy' not in track:
                    continue
                endpoint = (float(track['hn_endpoint_error_m'])
                            if method == 'hn' and 'hn_endpoint_error_m' in track else '')
                yield dict(**keys, method=method, reference_length_m=float(track['reference_length_m']),
                           predicted_length_m=track_length(track[f'{method}_xy']), endpoint_error_m=endpoint)


def from_table(path):
    for row in read_csv(path):
        if 'predicted_length_m' in row:
            yield {**{k: row[k] for k in ('cohort', 'sequence', 'route', 'mode', 'method',
                                          'reference_length_m', 'predicted_length_m')},
                   'endpoint_error_m': row.get('endpoint_error_m', '')}
        else:
            for method in ('hn', 'pdr'):
                key = f'{method}_length_m'
                if row.get(key):
                    yield dict(cohort=row.get('cohort', 'carrying'), sequence=row['sequence'],
                               route=row['route'], mode=row['mode'], method=method,
                               reference_length_m=row['reference_length_m'], predicted_length_m=row[key],
                               endpoint_error_m='')


def add_metrics(records):
    seen = set()
    for row in records:
        key = (row['cohort'], row['sequence'], row['method'])
        if key in seen:
            raise ValueError(f'Duplicate record: {key}')
        seen.add(key)
        predicted, reference = float(row['predicted_length_m']), float(row['reference_length_m'])
        if not (isfinite(predicted) and predicted >= 0 and isfinite(reference) and reference > 0):
            raise ValueError(f'Invalid predicted or reference path length for {key}.')
        tlr = predicted / reference
        yield dict(row, predicted_length_m=predicted, reference_length_m=reference, tlr=tlr,
                   signed_length_error_m=predicted - reference,
                   relative_path_length_error_pct=100 * abs(tlr - 1))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        base = (row['cohort'], row['method'])
        route, mode = row['route'], row['mode']
        for pair in (('All', 'All'), (route, 'All'), ('All', mode), (route, mode)):
            groups[base + pair].append(row)
    for (cohort, method, route, mode), group in sorted(groups.items()):
        endpoints = [float(r['endpoint_error_m']) for r in group if r['endpoint_error_m'] != '']
        yield dict(cohort=cohort, method=method, route=route, mode=mode, n=len(group),
                   mean_tlr=mean(r['tlr'] for r in group),
                   mean_relative_path_length_error_pct=mean(r['relative_path_length_error_pct'] for r in group),
                   mean_endpoint_error_m=mean(endpoints) if endpoints else '')


PAPER_METRICS = {'n': 0, 'mean_tlr': 3, 'mean_relative_path_length_error_pct': 2}


def paper_groups(summary):
    return {(r['route'], r['mode']): r for r in summary
            if r['method'] == 'hn' and r['cohort'] in ('carrying', 'handheld')}


def manuscript_table(summary):
    groups = paper_groups(summary)
    rows = []
    for route in ROUTES:
        for mode in ('Hand_Held', *PAPER_CARRYING_MODES):
            result = groups.get((route, mode))
            if result is None:
                raise ValueError(f'Table 2 requires HN results for {route}/{mode}.')
            rows.append(dict(route=route, mode=mode, **{
                field: f'{result[field]:.{digits}f}' for field, digits in PAPER_METRICS.items()
            }))
    return rows


def paper_check(summary, reference_path):
    group_index = paper_groups(summary)
    table = []
    reference_rows = read_csv(reference_path)
    for row in reference_rows:
        result = group_index.get((row['route'], row['mode']))
        if result is None:
            continue
        for field, digits in PAPER_METRICS.items():
            paper = row[field]
            computed = f'{result[field]:.{digits}f}'
            table.append(dict(route=row['route'], mode=row['mode'], metric=field,
                              paper=paper, recomputed=computed, match=(paper == computed)))
    report = dict(table2_expected_cells=len(reference_rows) * len(PAPER_METRICS),
                  table2_compared_cells=len(table), table2_matching_cells=sum(r['match'] for r in table))
    return report, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--tracks', type=Path, default=ROOT / 'results/trajectories')
    sources.add_argument('--length-table', type=Path, help='Path to an existing CSV of per-record path lengths.')
    parser.add_argument('--paper-only', action='store_true', help='Write manuscript Table 2 only.')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = list(add_metrics(from_table(args.length_table) if args.length_table else from_tracks(args.tracks)))
    if not rows:
        parser.error('The input contains no records to summarize.')
    summary = list(summarize(rows))
    with new_output(args.out) as out:
        if args.paper_only:
            write_csv(out / 'table2.csv', manuscript_table(summary))
        else:
            write_csv(out / 'per_record.csv', rows)
            write_csv(out / 'by_route_and_mode.csv', summary)
            reference_path = PAPER / 'table2.csv'
            if reference_path.is_file():
                report, table = paper_check(summary, reference_path)
                write_json(out / 'paper_check.json', report)
                write_csv(out / 'table2_comparison.csv', table,
                          ('route', 'mode', 'metric', 'paper', 'recomputed', 'match'))
    print(f'Summarized {len(rows)} method-record pairs in {args.out.resolve()}')


if __name__ == '__main__':
    main()
