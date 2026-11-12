"""Compute task-level KPIs, stop details, and scoring tables from cached HN-Transformer predictions."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from warehouse import picking
from warehouse.imu import load_input
from warehouse.protocol import (
    PAPER, PAPER_CARRYING_MODES, ROOT, ROUTES, check_sha256, new_output, read_csv, read_json, track_length, write_csv, write_json,
)
from warehouse.scoring import KPI, SCORES, score_profile, score_with_benchmarks


def evaluate(record, tracks):
    name, route = record['sequence'], record['route']
    input_path = ROOT / record['input']
    check_sha256(input_path, record['sha256'])
    time, features = load_input(input_path)
    path = tracks / f'{name}.npz'
    check_sha256(path, record['cache_sha256'])
    with np.load(path, allow_pickle=False) as track:
        ids, velocity = track['hn_indices'], track['hn_velocity_mps']
        length = float(track['reference_length_m'])
        summary, details, events, _ = picking.process_record(
            name, route, time, features, ids, velocity, length,
            picking.STATIONARY_DEFAULTS, picking.DEFAULTS)
        np.testing.assert_allclose(track_length(track['hn_xy']), summary[2], atol=1e-8, rtol=0)
    opt, appl, _, _, _, pdt, sst, tt, pending = summary[1:10]
    values = dict(zip(KPI, (opt, pdt, sst, tt, appl)))
    n_points = len(picking.PICKING_PERCENT[route])
    scores = score_profile(values, length, n_points)
    row = dict(sequence=name, route=route, cohort=record['cohort'], mode=record['mode'],
               personnel=record.get('personnel', ''), **values, **scores,
               reference_length_m=length, tlr=summary[3], total_dwell_s=summary[4],
               stop_count=summary[5], pending_dwell_s=pending, detector_edge_s=summary[10],
               classification_status=summary[11], expected_picking_points=n_points,
               matched_picking_points=summary[12])
    return row, details, events


def group_results(rows):
    personnel = defaultdict(list)
    carrying = defaultdict(list)
    for row in rows:
        if row['cohort'] == 'handheld' and row['route'] == 'Path3':
            personnel[row['personnel']].append(row)
        elif row['cohort'] == 'carrying':
            carrying[(row['route'], row['mode'])].append(row)
    profiles = []
    for name, group in sorted(personnel.items()):
        values = {key: mean(r[key] for r in group) for key in KPI}
        scores = score_profile(values, group[0]['reference_length_m'], group[0]['expected_picking_points'])
        profiles.append(dict(personnel=name, n=len(group), **values, **scores))
    return profiles, mean_condition_scores(carrying)


def mean_condition_scores(carrying):
    return [
        dict(route=route, mode=mode, n=len(group),
             **{key: mean(r[key] for r in group) for key in (*KPI, *SCORES)})
        for (route, mode), group in sorted(carrying.items())
    ]


def replay_paper_scoring():
    benchmarks = read_json(PAPER / 'scoring_benchmarks.json')
    profiles = []
    for row in read_csv(PAPER / 'table3_scoring_inputs.csv'):
        values = {key: float(row[key]) for key in KPI}
        scores = score_with_benchmarks(values, benchmarks['Path3'])
        profiles.append(dict(personnel=row['personnel'], **values, **scores))
    carrying = defaultdict(list)
    for row in read_csv(PAPER / 'table4_scoring_inputs.csv'):
        route, mode = row['route'], row['mode']
        values = {key: float(row[key]) for key in KPI}
        scores = score_with_benchmarks(values, benchmarks[route])
        carrying[(route, mode)].append(dict(**values, **scores))
    return profiles, mean_condition_scores(carrying)


def write_manuscript_tables(out, profiles, conditions):
    profile_index = {r['personnel']: r for r in profiles}
    condition_index = {(r['route'], r['mode']): r for r in conditions}
    fields = (*[field for pair in zip(KPI, SCORES[:-1], strict=True) for field in pair],
              'composite_score')
    table3 = [
        dict(personnel=name, **{field: f'{profile_index[name][field]:.2f}' for field in fields})
        for name in ('Personnel1', 'Personnel2', 'Personnel3', 'Personnel4')
    ]
    table4 = [
        dict(route=route, mode=mode, **{
            field: f'{condition_index[route, mode][field]:.2f}' for field in SCORES
        })
        for route in ROUTES for mode in PAPER_CARRYING_MODES
    ]
    write_csv(out / 'table3.csv', table3)
    write_csv(out / 'table4.csv', table4)


def compare(table, rows, keys, fields):
    reference_path = PAPER / f'{table}.csv'
    if not reference_path.is_file():
        return
    index = {tuple(row[k] for k in keys): row for row in rows}
    for paper in read_csv(reference_path):
        key = tuple(paper[k] for k in keys)
        if key not in index:
            continue
        computed = index[key]
        for field in fields:
            actual = f'{computed[field]:.2f}'
            yield dict(table=table, group='/'.join(key), metric=field, paper=paper[field],
                       recomputed=actual, difference=round(float(actual) - float(paper[field]), 2),
                       match=(actual == paper[field]))


def compare_paper_tables(profiles, conditions):
    return [*compare('table3', profiles, ('personnel',), (*KPI, *SCORES)),
            *compare('table4', conditions, ('route', 'mode'), SCORES)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', type=Path, default=ROOT / 'results/trajectories')
    parser.add_argument('--paper-only', action='store_true', help='Recalculate Tables 3 and 4 using the original workbook inputs and scoring thresholds.')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    replay_profiles, replay_conditions = replay_paper_scoring()
    if args.paper_only:
        with new_output(args.out) as out:
            write_manuscript_tables(out, replay_profiles, replay_conditions)
        print(f'Workbook-based scores written to {args.out.resolve()}')
        return
    manifest = read_json(args.tracks / 'manifest.json')
    expected = read_json(ROOT / 'config/inference.json')['checkpoint_sha256']
    if manifest['checkpoint_sha256'] != expected:
        raise ValueError('The trajectory cache was generated with a different checkpoint.')
    rows, stops, events = [], [], []
    for record in manifest['records']:
        row, details, groups = evaluate(record, args.tracks)
        rows.append(row)
        stops.extend(details)
        events.extend(groups)
    if not rows:
        parser.error('The trajectory cache contains no recordings.')
    profiles, conditions = group_results(rows)
    checks = compare_paper_tables(profiles, conditions)
    stop_fields = picking.DETAIL_COLUMNS + picking.DETAIL_EXTRA
    with new_output(args.out) as out:
        write_csv(out / 'per_task_kpis.csv', rows)
        write_csv(out / 'stop_details.csv',
                  (dict(zip(stop_fields, stop, strict=True)) for stop in stops), stop_fields)
        write_csv(out / 'picking_events.csv',
                  (dict(zip(picking.GROUP_COLUMNS, event, strict=True)) for event in events),
                  picking.GROUP_COLUMNS)
        write_csv(out / 'path3_by_personnel.csv', profiles, ('personnel', 'n', *KPI, *SCORES))
        write_csv(out / 'by_route_and_mode.csv', conditions, ('route', 'mode', 'n', *KPI, *SCORES))
        if checks:
            write_csv(out / 'paper_comparison.csv', checks)
        write_csv(out / 'paper_replay_table3.csv', replay_profiles)
        write_csv(out / 'paper_replay_table4.csv', replay_conditions)
        replay_checks = compare_paper_tables(replay_profiles, replay_conditions)
        if replay_checks:
            write_csv(out / 'paper_replay_comparison.csv', replay_checks)
        write_json(out / 'manifest.json', dict(
            task_count=len(rows), checkpoint_sha256=expected, stationary=picking.STATIONARY_DEFAULTS,
            matching=picking.DEFAULTS, picking_points_percent=picking.PICKING_PERCENT,
            task_mode='complete', table3='average KPIs, then score', table4='score tasks, then average'))
    print(f'Computed KPIs for {len(rows)} tasks in {args.out.resolve()}')


if __name__ == '__main__':
    main()
