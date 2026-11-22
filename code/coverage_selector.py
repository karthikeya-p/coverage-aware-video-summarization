"""Coverage-aware highlight selection (the core of this submission).

Action coverage (40 % of the score) counts *distinct* ground-truth verb-noun pairs that fall inside the
highlights under a 15 % time budget. We therefore pick highlights by maximising a monotone submodular
coverage function over *predicted* action instances:

    F(S) = sum_pairs  w_pair(S) * [pair newly covered]  +  REPEAT_W * (extra instances of covered pairs)
           + BETA * integral of action-ness over S

    s.t.  total length <= budget, every segment >= MIN_SEG, segments non-overlapping.

An instance counts as covered when >= 50 % of it lies inside S. Greedy cost-benefit (gain per second),
with the best-single-window fallback — the standard constant-factor approximation for budgeted
submodular maximisation (Khuller et al. 1999; Krause & Guestrin 2005).

The same `pack_windows` routine is reused by the FPVSum branch with a pure importance curve.
"""
import math, numpy as np
import config as C

RES = 0.1   # grid resolution in seconds


def _interp_curve(curve, t, duration):
    """Resample an (optional) importance/action-ness curve onto the RES grid in [0, duration]."""
    n = max(1, int(math.ceil(duration / RES)))
    grid = (np.arange(n) + 0.5) * RES
    if curve is None or len(curve) == 0:
        return np.zeros(n, np.float64), grid
    c = np.asarray(curve, np.float64)
    if t is None:
        t = np.linspace(0, duration, len(c))
    g = np.interp(grid, np.asarray(t, np.float64), c)
    lo, hi = np.percentile(g, 5), np.percentile(g, 95)
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
    return g, grid


def _finalize(windows, duration, budget, min_seg):
    """Sort, merge touching windows, clip, round, enforce strict non-overlap and the >=4 s rule."""
    ws = sorted([list(w) for w in windows])
    merged = []
    for s, e in ws:
        if merged and s <= merged[-1][1] + 0.2:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        s = max(0.0, round(s, 2)); e = min(duration, round(e, 2))
        if out and s <= out[-1][1]:
            s = round(out[-1][1] + 0.01, 2)
        if e - s >= min(min_seg, duration) - 1e-9:
            out.append([s, e])
    if not out:  # degenerate: one window of min_seg at the best place we can (start)
        out = [[0.0, round(min(duration, min_seg), 2)]]
    # trim total length to budget (never below one min_seg window)
    total = sum(e - s for s, e in out)
    i = len(out) - 1
    while total > budget + 1e-6 and len(out) > 1:
        excess = total - budget
        s, e = out[i]
        cut = min(excess, (e - s) - min_seg)
        if cut > 0:
            out[i][1] = round(e - cut, 2); total -= cut
        else:
            total -= (e - s); out.pop(i)
        i -= 1
        if i < 0:
            break
    return out


def pack_windows(curve, t, duration, budget_frac=C.BUDGET_FRAC, min_seg=C.MIN_SEG, max_seg=C.MAX_SEG):
    """Pure importance knapsack: choose non-overlapping windows (len in [min_seg, max_seg]) maximising
    the integral of the curve under the budget. Greedy by mean importance with window growth."""
    g, grid = _interp_curve(curve, t, duration)
    budget = max(budget_frac * duration, min(min_seg, duration))
    n = len(g)
    cs = np.concatenate([[0.0], np.cumsum(g)]) * RES
    occ = np.zeros(n, bool)
    chosen, used = [], 0.0
    L = max(1, int(round(min_seg / RES)))
    while used + min_seg <= budget + 1e-9:
        # mean importance of every min_seg window that is free
        sums = cs[L:] - cs[:-L]                       # window starting at i covers [i, i+L)
        free = np.ones(n - L + 1, bool)
        occ_cs = np.concatenate([[0], np.cumsum(occ)])
        free &= (occ_cs[L:] - occ_cs[:-L]) == 0
        if not free.any():
            break
        sums = np.where(free, sums, -np.inf)
        i = int(sums.argmax())
        if not np.isfinite(sums[i]):
            break
        s, e = i, i + L
        # grow while the next cell is free, above the window mean, within max_seg and the budget
        while (e - s) * RES < max_seg and used + (e - s + 1) * RES <= budget + 1e-9:
            left = g[s - 1] if s > 0 and not occ[s - 1] else -1
            right = g[e] if e < n and not occ[e] else -1
            mean = (cs[e] - cs[s]) / ((e - s) * RES)
            if max(left, right) < 0.6 * mean:
                break
            if right >= left:
                e += 1
            else:
                s -= 1
        occ[s:e] = True
        chosen.append([s * RES, e * RES]); used += (e - s) * RES
    return _finalize(chosen, duration, budget, min_seg)


def select_highlights(actions, duration, actionness=None, act_t=None, budget_frac=C.BUDGET_FRAC,
                      min_seg=C.MIN_SEG, max_seg=C.SEL_MAX_SEG, pad=C.SEL_PAD, beta=C.SEL_BETA,
                      repeat_w=C.SEL_REPEAT_WEIGHT, min_score=C.SEL_MIN_SCORE, return_stats=False):
    """Coverage-aware selection over predicted action instances (see module docstring)."""
    g, grid = _interp_curve(actionness, act_t, duration)
    budget = max(budget_frac * duration, min(min_seg, duration))
    sc = lambda a: a.get("sel_score", a["score"])          # detector score x co-occurrence prior when available
    inst = [a for a in actions if sc(a) >= min_score]
    inst.sort(key=lambda a: -sc(a))
    inst = inst[:3000]
    if not inst:
        return (pack_windows(actionness, act_t, duration, budget_frac, min_seg, max_seg), {}) if return_stats \
            else pack_windows(actionness, act_t, duration, budget_frac, min_seg, max_seg)

    S = np.array([a["segment"][0] for a in inst]); E = np.array([a["segment"][1] for a in inst])
    W = np.array([sc(a) for a in inst]); LEN = np.maximum(E - S, 0.1)
    pair_keys = [(a["verb"], a["noun"]) for a in inst]
    uniq = sorted(set(pair_keys)); pid = {k: i for i, k in enumerate(uniq)}
    P = np.array([pid[k] for k in pair_keys]); n_pairs = len(uniq)
    order = np.argsort(P, kind="stable")                       # instances grouped by pair
    P_sorted = P[order]; starts = np.flatnonzero(np.r_[True, P_sorted[1:] != P_sorted[:-1]])

    # ---- candidate windows: one per instance (padded, >= min_seg) + an action-ness grid
    cands = []
    for s, e in zip(S, E):
        L = min(max_seg, max(min_seg, (e - s) + 2 * pad)); c = 0.5 * (s + e)
        a = max(0.0, c - L / 2); b = a + L
        if b > duration:
            b = duration; a = max(0.0, b - L)
        cands.append((a, b))
    for a in np.arange(0.0, max(duration - min_seg, 0) + 1e-9, 1.0):
        cands.append((a, min(duration, a + min_seg)))
    CA = np.array([c[0] for c in cands]); CB = np.array([c[1] for c in cands]); CL = CB - CA
    cs_g = np.concatenate([[0.0], np.cumsum(g)]) * RES

    def act_mass(a, b):
        i, j = int(a / RES), min(len(g), int(math.ceil(b / RES)))
        return cs_g[j] - cs_g[i]

    # overlap of every candidate with every instance (seconds)
    O = np.maximum(0.0, np.minimum(CB[:, None], E[None, :]) - np.maximum(CA[:, None], S[None, :]))
    inside = np.zeros(len(inst))                               # seconds of each instance already selected
    pair_cov = np.zeros(n_pairs, bool)
    alive = np.ones(len(cands), bool)
    chosen, used = [], 0.0
    stats = dict(n_instances=len(inst), n_pairs=n_pairs, steps=[])

    while True:
        fits = alive & (CL <= budget - used + 1e-9)
        if not fits.any():
            break
        idx = np.flatnonzero(fits)
        newly = (inside[None, :] + O[idx] >= 0.5 * LEN[None, :]) & (inside[None, :] < 0.5 * LEN[None, :])
        val = np.where(newly, W[None, :], 0.0)                 # cand x inst
        # per-pair max over instances -> cand x pairs
        pair_val = np.maximum.reduceat(val[:, order], starts, axis=1)
        gain = (pair_val * np.where(pair_cov, repeat_w, 1.0)[None, :]).sum(1)
        gain += beta * np.array([act_mass(CA[i], CB[i]) for i in idx])
        ratio = gain / CL[idx]
        k = int(ratio.argmax())
        if gain[k] <= 1e-9:
            break
        ci = idx[k]
        a, b = CA[ci], CB[ci]
        chosen.append((a, b)); used += CL[ci]
        inside += O[ci]
        pair_cov = np.maximum.reduceat((inside >= 0.5 * LEN)[order].astype(np.int8), starts).astype(bool)
        # kill overlapping candidates (we only ever add disjoint windows)
        alive &= ~((CA < b) & (CB > a))
        stats["steps"].append(dict(window=[round(a, 2), round(b, 2)], gain=round(float(gain[k]), 4),
                                   pairs_covered=int(pair_cov.sum())))

    # spend leftover budget (< min_seg) by widening chosen windows toward unused action mass
    left = budget - used
    if chosen and left > 0.3:
        chosen = sorted(chosen)
        share = left / len(chosen)
        widened = []
        for i, (a, b) in enumerate(chosen):
            lo = widened[-1][1] if widened else 0.0
            hi = chosen[i + 1][0] if i + 1 < len(chosen) else duration
            la = act_mass(max(lo, a - share), a); ra = act_mass(b, min(hi, b + share))
            if ra >= la:
                b2 = min(hi, b + share); a2 = max(lo, a - (share - (b2 - b)))
            else:
                a2 = max(lo, a - share); b2 = min(hi, b + (share - (a - a2)))
            widened.append([a2, b2])
        chosen = widened

    out = _finalize(chosen, duration, budget, min_seg)
    if return_stats:
        covered_pairs = int(pair_cov.sum())
        stats.update(pairs_covered=covered_pairs, total_len=round(sum(e - s for s, e in out), 2),
                     budget=round(budget, 2))
        return out, stats
    return out
