"""Published project’s stationary accounting and ordered picking-point matching."""
import numpy as np

from .imu import integrate, stationary_intervals


PICKING_PERCENT = {
    'Path1': [5.02, 15.06, 21.75, 30.67, 40.52, 47.03, 78.76, 81.83],
    'Path2': [9.98, 21.83, 31.81, 46.93, 50.05, 87.79, 91.32],
    'Path3': [11.18, 21.12, 39.13, 67.24, 71.58, 88.66, 91.56],
}


SUMMARY_COLUMNS = [
    'sequence', 'opt_s', 'appl_m', 'TLR', 'total_dwell_s', 'stop_count',
    'pdt_s', 'sst_s', 'tt_s',
    'pending_dwell_s', 'detector_edge_s', 'classification_status',
]


DETAIL_COLUMNS = [
    'sequence', 'stop_id', 'start_time_s', 'end_time_exclusive_s', 'start_elapsed_s',
    'end_elapsed_s', 'dwell_s', 'predicted_route_progress_pct', 'nearest_pick_point_id',
    'nearest_pick_point_progress_pct', 'nearest_point_error_pp', 'classification', 'trajectory_overlap_s',
]


def summarize(name, route, time, labels, intervals, tp, xy, reference_length,
              tolerance_pp=.5):
    """Account for the full recording and classify stops with complete trajectory coverage."""
    if not np.isfinite(tolerance_pp) or not 0 <= tolerance_pp <= 100:
        raise ValueError('The matching tolerance must be finite and between 0 and 100 percentage points.')
    if route not in PICKING_PERCENT:
        raise ValueError('The route must be Path1, Path2, or Path3.')
    if (tp.ndim != 1 or len(tp) < 2 or xy.shape != (len(tp), 2)
            or not np.isfinite(tp).all() or not np.isfinite(xy).all()
            or np.any(np.diff(tp) <= 0)):
        raise ValueError('The predicted trajectory must contain at least two finite points with strictly increasing timestamps.')
    if reference_length is not None and (
            not np.isfinite(reference_length) or reference_length <= 0):
        raise ValueError('The reference path length must be finite and positive.')
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if not np.isfinite(cumulative).all():
        raise ValueError('Cumulative predicted path length contains nonfinite values.')
    length = float(cumulative[-1])
    total = float(time[-1] - time[0])
    dwell = picking = other = pending = 0.
    details = []
    previous_end = float(time[0])
    targets = np.asarray(PICKING_PERCENT[route])
    for number, (start, end) in enumerate(intervals, 1):
        if (not np.isfinite([start, end]).all() or start < previous_end - 1e-9
                or end <= start or end > time[-1] + 1e-9):
            raise ValueError('Stop intervals must be ordered, nonoverlapping, and within the recording.')
        previous_end = end
        duration = float(end - start)
        dwell += duration
        overlap = max(0., min(end, tp[-1]) - max(start, tp[0]))
        position = point_id = target = error = None
        category = 'Review required (incomplete trajectory coverage or zero path length)'
        if start >= tp[0] - 1e-9 and end <= tp[-1] + 1e-9 and length > 1e-9:
            # Use the nearest route-progress target as a preliminary stop classification.
            samples = time[(time >= start) & (time < end)]
            if not len(samples):
                raise ValueError('The stop interval contains no sensor samples.')
            position = float(np.median(np.interp(samples, tp, cumulative)) / length * 100.)
            nearest = int(np.argmin(np.abs(targets - position)))
            point_id, target = nearest + 1, float(targets[nearest])
            error = abs(position - target)
            if error <= tolerance_pp + 1e-9:
                category = 'Picking-point dwell'
                picking += duration
            else:
                category = 'Non-picking-point dwell'
                other += duration
        else:
            pending += duration
        details.append([name, number, float(start), float(end), float(start - time[0]),
                        float(end - time[0]), duration, position, point_id, target,
                        error, category, float(overlap)])
    walk = total - dwell
    if walk < -1e-8 or not np.isclose(dwell, picking + other + pending, atol=1e-8, rtol=0):
        raise ValueError('The time components do not add up.')
    edge_time = float(np.diff(time)[labels[:-1] == -1].sum())
    row = [name, total, length, length / reference_length if reference_length else 'NA',
           dwell, len(intervals), picking, other, max(0., walk), pending, edge_time,
           'Some stops require review' if pending else 'All detected stops classified']
    return row, details


STATIONARY_DEFAULTS = dict(window=.5, min_duration=1., gyro_threshold=.4,
                           acc_threshold=.5, gravity=9.81, exit_factor=1.5)


DEFAULTS = dict(tolerance_pp=.5, position_scale_pp=2., position_weight=.25,
                skip_stop_penalty=1., miss_point_penalty=2., merge_penalty=.2,
                merge_gap_s=2., merge_distance_m=.5, min_margin=.25)


SUMMARY_EXTRA = ['matched_picking_points', 'confirmed_picking_events', 'merged_confirmed_segments',
                 'unassigned_pick_point_ids', 'matching_mode', 'stop_position_warning_count', 'total_matching_cost']


DETAIL_EXTRA = ['nearest_point_classification', 'assigned_pick_point_id', 'assigned_pick_point_progress_pct',
                'assigned_point_error_pp', 'event_id', 'assignment_method', 'position_warning',
                'alternative_cost_margin']


GROUP_COLUMNS = ['sequence', 'event_id', 'pick_point_id', 'source_stop_ids',
                 'first_stop_start_s', 'last_stop_end_exclusive_s',
                 'total_detected_dwell_s', 'total_between_stop_gap_s',
                 'duration_weighted_route_progress_pct', 'classification', 'max_position_error_pp']


def validate_options(options):
    p = dict(DEFAULTS, **options)
    if set(p) != set(DEFAULTS) or not np.isfinite(list(p.values())).all():
        raise ValueError('Matching options must have recognized names and finite values.')
    if (min(p.values()) < 0 or p['position_scale_pp'] <= 0
            or min(p['skip_stop_penalty'], p['miss_point_penalty']) <= 0
            or p['tolerance_pp'] > 100):
        raise ValueError('Options must be nonnegative. The position scale and skip/miss penalties must be positive. The warning tolerance must be at most 100.')
    return p


def match_sequence(positions, targets, durations=None, merge_links=None,
                   task_mode='complete', **options):
    """Assign stops to ordered picking points by dynamic programming.

    DP(i, j) accounts for i stop intervals and j picking points. It can skip a
    stop, assign one or more consecutive stops, or leave a point unassigned.
    Complete mode requires full coverage when enough valid stops are present.
    Missing positions remain pending review. The merge_links array specifies
    which adjacent intervals can be assigned to the same event.
    """
    p = validate_options(options)
    if task_mode not in ('complete', 'partial'):
        raise ValueError('task_mode must be complete or partial.')
    r = np.asarray([np.nan if v is None else v for v in positions], float)
    q = np.asarray(targets, float)
    n, m = len(r), len(q)
    valid = ~np.isnan(r)
    weights = np.asarray(np.ones(n) if durations is None else durations, float)
    links = np.asarray([False] * max(0, n - 1) if merge_links is None else merge_links)
    if not links.size:
        links = np.asarray([], dtype=bool)
    if (r.ndim != 1 or q.ndim != 1 or not m or not np.isfinite(q).all()
            or not np.isfinite(r[valid]).all() or np.any((r[valid] < 0) | (r[valid] > 100))
            or np.any(np.diff(r[valid]) < -1e-9) or np.any(np.diff(q) <= 0)
            or np.any((q < 0) | (q > 100)) or weights.shape != (n,)
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or links.shape != (max(0, n - 1),) or links.dtype != bool):
        raise ValueError('Positions must be finite and within [0, 100]. Stops must be ordered and targets strictly increasing. Durations must be positive and merge flags must be Boolean.')
    # Only explicit None values represent missing positions.
    if any(v is not None and not np.isfinite(v) for v in positions):
        raise ValueError('Use None for missing positions. Other positions must be finite.')
    allow_missing = task_mode == 'partial' or int(valid.sum()) < m
    mode = ('Unassigned points allowed' if task_mode == 'partial' else
            'Insufficient stops (unassigned points allowed)' if allow_missing else 'Complete task order')
    blocks = [[] for _ in range(n)]
    for start in range(n):
        if not valid[start]:
            continue
        for end in range(start + 1, n + 1):
            if end > start + 1 and (not valid[end - 1] or not links[end - 2]):
                break
            residual = np.abs((r[start:end, None] - q) / p['position_scale_pp'])
            huber = np.where(residual <= 1., .5 * residual ** 2, residual - .5)
            costs = (p['position_weight'] * np.average(huber, axis=0, weights=weights[start:end])
                     + p['merge_penalty'] * (end - start - 1))
            blocks[start].append((end, costs))

    def solve(forbidden=None, required_stop=None):
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.
        parent = {}

        def relax(i, j, end, point, value, action):
            value += cost[i, j]
            if value < cost[end, point] - 1e-12:
                cost[end, point] = value
                parent[end, point] = (i, j, action)

        for i in range(n + 1):
            for j in range(m + 1):
                if not np.isfinite(cost[i, j]):
                    continue
                if i < n and i != required_stop:
                    relax(i, j, i + 1, j, p['skip_stop_penalty'] if valid[i] else 0., 'skip')
                if j < m and allow_missing:
                    relax(i, j, i, j + 1, p['miss_point_penalty'], 'miss')
                if i < n and j < m:
                    for end, values in blocks[i]:
                        if forbidden is not None and j == forbidden[1] and i <= forbidden[0] < end:
                            continue
                        relax(i, j, end, j + 1, values[j], 'pick')
        groups, state = [], (n, m)
        if np.isfinite(cost[state]):
            while state != (0, 0):
                i, j, action = parent[state]
                if action == 'pick':
                    groups.append((i, state[0], j))
                state = (i, j)
        return float(cost[n, m]), groups[::-1]

    # For these short tasks, solve again to measure each alternative assignment cost.
    score, groups = solve()
    result = [dict(point=None, group=None, margin=None,
                   category='Non-picking-point dwell' if valid[i] else 'Review required (incomplete position)') for i in range(n)]
    events = []
    for number, (start, end, j) in enumerate(groups, 1):
        margins = []
        for i in range(start, end):
            alternative, _ = solve(forbidden=(i, j))
            margins.append(max(0., alternative - score) if np.isfinite(alternative) else None)
        ambiguous = any(v is not None and v + 1e-9 < p['min_margin'] for v in margins)
        category = 'Review required (ambiguous point assignment)' if ambiguous else 'Picking-point dwell (task order)'
        for i, margin in zip(range(start, end), margins):
            result[i] = dict(point=j + 1, group=number, margin=margin, category=category)
        events.append(dict(start=start, end=end, point=j + 1, number=number,
                           category=category,
                           position=float(np.average(r[start:end], weights=weights[start:end]))))
    for i, item in enumerate(result):
        if valid[i] and item['point'] is None:
            alternative, _ = solve(required_stop=i)
            if np.isfinite(alternative):
                item['margin'] = max(0., alternative - score)
                if item['margin'] + 1e-9 < p['min_margin']:
                    item['category'] = 'Review required (ambiguous extra stop)'
    missing = sorted(set(range(1, m + 1)) - {g['point'] for g in events})
    return result, events, dict(score=score, mode=mode, missing_points=missing)


def refine_record(summary, details, targets, tp, xy, task_mode='complete', **options):
    p = validate_options(options)
    if (len(details) != summary[5] or not np.isclose(summary[1], summary[4] + summary[8])
            or not np.isclose(summary[4], sum(d[6] for d in details))
            or not np.isclose(summary[4], summary[6] + summary[7] + summary[9])):
        raise ValueError('The source stop details and time totals are inconsistent.')
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    links = []
    for left, right in zip(details, details[1:]):
        gap = right[2] - left[3]
        if gap < -1e-9:
            raise ValueError('Stop intervals overlap or are out of order.')
        distance = float(np.interp(right[2], tp, cumulative) - np.interp(left[3], tp, cumulative))
        links.append(bool(left[7] is not None and right[7] is not None
                          and gap <= p['merge_gap_s'] + 1e-9
                          and distance <= p['merge_distance_m'] + 1e-9))
    matches, events, info = match_sequence([d[7] for d in details], targets,
                                          [d[6] for d in details], links,
                                          task_mode=task_mode, **p)
    refined, warnings = [], 0
    for d, match in zip(details, matches):
        point = match['point']
        target = targets[point - 1] if point is not None else None
        error = abs(d[7] - target) if target is not None else None
        warning = 'Position error exceeds warning tolerance' if error is not None and error > p['tolerance_pp'] + 1e-9 else ''
        warnings += bool(warning)
        source = ('Task order (complete task assumed)' if info['mode'] == 'Complete task order'
                  else 'Task order and position (unassigned points allowed)') if point is not None else 'Unassigned'
        row = list(d) + [d[11], point, target, error, match['group'], source, warning, match['margin']]
        row[11] = match['category']
        refined.append(row)
    group_rows = []
    for event in events:
        rows = refined[event['start']:event['end']]
        dwell = sum(d[6] for d in rows)
        group_rows.append([summary[0], event['number'], event['point'],
                           ','.join(str(d[1]) for d in rows), rows[0][2], rows[-1][3], dwell,
                           max(0., rows[-1][3] - rows[0][2] - dwell), event['position'],
                           event['category'], max(d[16] for d in rows)])
    updated = list(summary)
    updated[6] = sum(d[6] for d in refined if d[11].startswith('Picking-point dwell'))
    updated[7] = sum(d[6] for d in refined if d[11] == 'Non-picking-point dwell')
    updated[9] = sum(d[6] for d in refined if d[11].startswith('Review required'))
    if not np.isclose(updated[4], updated[6] + updated[7] + updated[9], atol=1e-8, rtol=0):
        raise ValueError('The time components do not add up after ordered matching.')
    updated[11] = ('Some stops require review' if updated[9] else
                   'Some picking points are unassigned' if info['missing_points'] else
                   'Assigned by task order with position warnings' if warnings else 'All detected stops classified')
    confirmed = [g for g in events if g['category'].startswith('Picking-point dwell')]
    updated.extend([len(confirmed), len(confirmed), sum(g['end'] - g['start'] - 1 for g in confirmed),
                    ','.join(map(str, info['missing_points'])) or '', info['mode'], warnings, info['score']])
    return updated, refined, group_rows, info


def process_record(name, route, time, feat, ids, velocity, reference_length,
                   stationary_settings, matching_options, task_mode='complete'):
    if (ids is None or len(ids) < 2 or velocity is None or velocity.shape != (len(ids), 2)
            or not np.isfinite(velocity).all()):
        raise ValueError(name + ': too few valid velocity predictions.')
    labels, intervals = stationary_intervals(time, feat, **stationary_settings)
    tp, xy = integrate(time, ids, velocity)
    summary, details = summarize(name, route, time, labels, intervals, tp, xy,
                                   reference_length, matching_options['tolerance_pp'])
    return refine_record(summary, details, PICKING_PERCENT[route], tp, xy,
                         task_mode=task_mode, **matching_options)
