"""Warehouse manuscript Table 1 efficacy-coefficient scoring."""
from math import ceil

KPI = ('opt_s', 'pdt_s', 'sst_s', 'tt_s', 'appl_m')
SCORES = ('opt_score', 'pdt_score', 'sst_score', 'tt_score', 'appl_score', 'composite_score')
WEIGHTS = (.1, .3, .4, .1, .1)


def score_profile(profile, length, points):
    allowance = ceil(length / 10)
    satisfied = (length / 1.2 + 8 * points, 8 * points, 0, length / 1.2, length)
    threshold = (length / .8 + 12 * points + allowance, 12 * points, allowance, length / .8, 1.1 * length)
    return score_with_benchmarks(profile, zip(satisfied, threshold))


def score_with_benchmarks(profile, benchmarks):
    scores = [min(100., max(0., 60 + 40 * (high - profile[key]) / (high - low)))
              for key, (low, high) in zip(KPI, benchmarks, strict=True)]
    composite = sum(weight * score for weight, score in zip(WEIGHTS, scores))
    return dict(zip(SCORES, [*scores, composite]))
