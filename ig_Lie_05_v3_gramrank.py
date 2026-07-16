from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
Lie rad 05 v3 — Gramijan / adjoint-rank → next

POPRAVKA: sirovi span(Xp,…) gurаo sve u low zonu.
  1) unit-norm vi pre Gram-a (X ne dominira)
  2) debias s unutar zone (s − mean_zone)
  3) combo_fit: kazna za neuravnotežene zone

Generatori X (zone), Y (par/nepar), Z=[X,Y].
Gram G_ij=<vi,vj>; rank_eff; skor iz span + excess.
Ban last; next. CSV ceo, seed=39.
Ime: ig_Lie_05_v3_gramrank.py
"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
WINDOW = 100
RANK_EPS = 1e-6
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def zone(n: int) -> int:
    if n <= 13:
        return 0
    if n <= 26:
        return 1
    return 2


def window_p(draws: np.ndarray, end: int, w: int = WINDOW) -> np.ndarray:
    start = max(0, end - w)
    chunk = draws[start:end]
    cnt = Counter(chunk.reshape(-1).tolist())
    n_slots = max(len(chunk) * FRONT_SELECT, 1)
    p = np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def global_p(draws: np.ndarray) -> np.ndarray:
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    p = np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)
    return p / p.sum()


def zone_cycle_generator() -> np.ndarray:
    members = [[], [], []]
    for i in range(FRONT_N):
        members[zone(i + 1)].append(i)
    X = np.zeros((FRONT_N, FRONT_N))
    for a in range(3):
        b = (a + 1) % 3
        Za, Zb = members[a], members[b]
        na, nb = float(len(Za)), float(len(Zb))
        for i in Za:
            for j in Zb:
                X[j, i] += 1.0 / (na * nb)
                X[i, j] -= 1.0 / (na * nb)
    return X


def parity_generator() -> np.ndarray:
    odd = [i for i in range(FRONT_N) if (i + 1) % 2 == 1]
    even = [i for i in range(FRONT_N) if (i + 1) % 2 == 0]
    Y = np.zeros((FRONT_N, FRONT_N))
    no, ne = float(len(odd)), float(len(even))
    for i in odd:
        for j in even:
            Y[j, i] += 1.0 / (no * ne)
            Y[i, j] -= 1.0 / (no * ne)
    return Y


def project_tangent(v: np.ndarray) -> np.ndarray:
    return v - v.mean()


def unit(v: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 1e-18 else v


def lie_velocities(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    X = zone_cycle_generator()
    Y = parity_generator()
    Z = X @ Y - Y @ X
    # unit-norm: jednaki doprinos pravaca (inače X gura low)
    v1 = unit(project_tangent(X @ p))
    v2 = unit(project_tangent(Y @ p))
    v3 = unit(project_tangent(Z @ p))
    V = np.column_stack([v1, v2, v3])
    Gram = V.T @ V
    w, _ = np.linalg.eigh(Gram)
    w = np.clip(w, 0.0, None)
    wmax = float(w.max()) if w.max() > 0 else 1.0
    rank_eff = int(np.sum(w > RANK_EPS * wmax))
    return v1, v2, v3, rank_eff, Gram


def span_score(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, Gram: np.ndarray) -> np.ndarray:
    try:
        alpha = np.linalg.pinv(Gram, rcond=RANK_EPS) @ np.ones(3)
    except np.linalg.LinAlgError:
        alpha = np.ones(3) / 3.0
    s = alpha[0] * v1 + alpha[1] * v2 + alpha[2] * v3
    # debias po zoni — sklanja sistemski low/mid/high bias
    for z in range(3):
        idx = [i for i in range(FRONT_N) if zone(i + 1) == z]
        if not idx:
            continue
        mu = float(s[idx].mean())
        s[idx] = s[idx] - mu
    return s


def number_scores(
    s: np.ndarray, p_now: np.ndarray, p_glob: np.ndarray, ban: set[int]
) -> dict[int, float]:
    out = {}
    for i in range(FRONT_N):
        n = i + 1
        if n in ban:
            out[n] = -1e18
        else:
            out[n] = float(s[i] + 0.15 * (p_now[i] - p_glob[i]))
    return out


def _zone_counts(nums: list[int]) -> list[int]:
    c = [0, 0, 0]
    for x in nums:
        c[zone(x)] += 1
    return c


def _combo_fit(combo, score, target_sum, pos_means, target_odd, ban):
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    # cilj ~2/2/3 ili 3/2/2 — kazni monozonu
    zc = _zone_counts(nums)
    s -= 0.55 * sum(abs(c - FRONT_SELECT / 3.0) for c in zc)
    if max(zc) >= 5:
        s -= 2.0 * (max(zc) - 4)
    return s


def predict_next(draws, score, ban):
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))
    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))
    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_ig_05_v3(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    n = len(draws)
    p_now = window_p(draws, n, WINDOW)
    p_glob = global_p(draws)
    v1, v2, v3, rank_eff, Gram = lie_velocities(p_now)
    s = span_score(v1, v2, v3, Gram)
    score = number_scores(s, p_now, p_glob, ban)
    combo = predict_next(draws, score, ban)

    ev = np.linalg.eigvalsh(Gram)
    print(f"CSV: {csv_path.name}")
    print(f"Kola: {n} | seed={SEED} | W={WINDOW} | ig_Lie_05_v3 gramrank")
    print(f"last: {last.tolist()}")
    print()
    print("=== Gram / rank ===")
    print(
        {
            "rank_eff": rank_eff,
            "eig": [round(float(x), 8) for x in ev.tolist()],
            "||span||": round(float(np.linalg.norm(s)), 6),
        }
    )
    print()
    ranked = sorted(
        ((n_, float(score[n_])) for n_ in range(1, FRONT_N + 1) if n_ not in ban),
        key=lambda t: (-t[1], t[0]),
    )
    print("=== top12 skor (span debiased) ===")
    print([(n_, round(sc, 6)) for n_, sc in ranked[:12]])
    print()
    print("=== next (ig_Lie_05_v3 gramrank) ===")
    print("next:", combo)


if __name__ == "__main__":
    run_ig_05_v3()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | W=100 | ig_Lie_05_v3 gramrank
last: [4, 5, 6, 11, 12, 18, 28]

=== Gram / rank ===
{'rank_eff': 3, 'eig': [0.00108813, 1.0, 1.99891187], '||span||': 2.106767}

=== top12 skor (span debiased) ===
[(19, 0.364689), (17, 0.364062), (15, 0.363682), (25, 0.363498),(23, 0.363413), (21, 0.362929), (1, 0.31445), (29, 0.31375), (27, 0.313393), (31, 0.31277), (9, 0.312501), (7, 0.312482)]

=== next (ig_Lie_05_v3 gramrank) ===
next: [7, 9, 15, 21, 27, 29, 31]
"""



"""
adjoint orbita / Gramijan kontrolabilnosti (rank span{Xp, Yp, [X,Y]p}) → skor.

Gramijan na {Xp, Yp, [X,Y]p} → rank_eff + skor iz span-a.

sprečavam kolaps u low zonu (normalizacija generatora + zona-balans u skoru/combo_fit).
Korekcija: unit-norm vi + debias po zoni + kazna monozone u combo_fit.
"""



"""
Rad	    Tema
03      Villani / Optimal Transport
04      Monge controller
05      Lie groups + observability





#	Jezgro	Fajlovi

01 - 6
Stošić / Fisher
ig_Stosic_01_fisher_v1_empfreq … v6_banlast

02 - 23
Stošić / IG (chart–Γ–ekscitacija–χ–τ–…)
ig_Stosic_02_v1_multichart … v23_permnull

03 - 3
Villani / OT
ig_Villani_03_v1_sinkhorn, v2_w1path, v3_displacement

04 - 2
Monge kontroler
ig_Monge_04_v1_map, v2_iterctrl

05 - 3
Lie + observability
ig_Lie_05_v1_genobs, v2_bracket, v3_gramrank
"""






"""
01 - 6   02 - 23   03 - 3   04 - 2   05 - 3 
37 kombinacija

v11 ima dve (BRIGHT / CCT)

ukupno: 38 kombinacija
"""



"""
37 next (seed=39, CSV loto7_4650_k56):

01 Fisher (6)
v1_empfreq — [9, 10, 16, 21, 24, 25, 35]
v2_jaccard — [5, 8, 13, 23, 26, 30, 38]
v3_meancarrier — [5, 12, 13, 21, 27, 30, 32]
v4_maxcarrier — [5, 9, 15, 21, 24, 30, 36]
v5_pairs — [5, 12, 14, 21, 23, 30, 35]
v6_banlast — [9, 10, 16, 19, 25, 29, 32]

02 Stošić (23)
v1_multichart — [9, 10, 16, 19, 25, 29, 32]
v2_christoffel — [3, 9, 14, 22, 27, 30, 35]
v3_hysteresis — [9, 10, 16, 19, 25, 29, 32]
v4_naturalgradient — [9, 10, 15, 16, 25, 29, 36]
v5_geodesic — [7, 9, 19, 20, 24, 30, 31]
v6_dualem — [8, 9, 15, 20, 24, 31, 34]
v7_klfisherrao — [3, 13, 17, 20, 22, 30, 35]
v8_atlasab — [2, 9, 14, 19, 25, 33, 36]
v9_microexcitation — [2, 13, 15, 19, 25, 30, 36]
v10_stdgeoab — [9, 10, 15, 16, 25, 29, 36]
v11_excitationmodes — BRIGHT [2, 13, 15, 19, 26, 29, 36] / CCT [3, 8, 13, 19, 29, 32, 36]
v12_combinedexcitation — [8, 9, 16, 17, 25, 30, 35]
v13_abphases — [9, 10, 15, 16, 25, 29, 36]
v14_tangentrotation — [3, 9, 14, 21, 26, 30, 37]
v15_spectralpersistence — [8, 9, 16, 17, 25, 30, 35]
v16_lambdagate — [9, 10, 15, 16, 25, 29, 36]
v17_fisherrank — [3, 13, 14, 21, 24, 31, 34]
v18_chiaxis — [14, 15, 16, 19, 21, 24, 31]
v19_taudrift — [2, 14, 21, 23, 25, 26, 30]
v20_levicivitareg — [2, 10, 15, 21, 25, 31, 36]
v21_lagxcorr — [1, 13, 14, 17, 26, 33, 36]
v22_temporalprecedence — [3, 10, 13, 22, 25, 32, 35]
v23_permnull — [1, 14, 16, 19, 27, 29, 34]

03 Villani (3)
v1_sinkhorn — [8, 9, 15, 20, 24, 31, 34]
v2_w1path — [7, 8, 15, 19, 27, 30, 34]
v3_displacement — [3, 9, 19, 20, 24, 31, 34]

04 Monge (2)
v1_map — [2, 9, 14, 19, 25, 33, 36]
v2_iterctrl — [2, 9, 19, 21, 26, 30, 35]

05 Lie (3)
v1_genobs — [7, 8, 9, 13, 32, 36, 37]
v2_bracket — [1, 8, 10, 27, 29, 31, 34]
v3_gramrank — [7, 9, 15, 21, 27, 29, 31]

Napomena: v11 ima dve (BRIGHT / CCT); ostalih 36 po jednu.

37 skripti, 38 kombinacija 
"""



"""
Frekvencija po 38 kombinacija (266 mesta), veće → manje:

broj	puta	udeo
9
22
8.3%
25
16
6.0%
19
14
5.3%
30
14
5.3%
15
13
4.9%
36
13
4.9%
16
12
4.5%
29
12
4.5%
10
11
4.1%
21
11
4.1%
13
10
3.8%
14
10
3.8%
8
9
3.4%
31
9
3.4%
24
8
3.0%
35
8
3.0%
2
7
2.6%
3
7
2.6%
32
7
2.6%
34
7
2.6%
26
6
2.3%
27
6
2.3%
20
5
1.9%
5
4
1.5%
7
4
1.5%
17
4
1.5%
1
3
1.1%
22
3
1.1%
23
3
1.1%
33
3
1.1%
12
2
0.8%
37
2
0.8%
38
1
0.4%


0×: 4, 6, 11, 18, 28, 39
(4–28 su last → ban; 39 se nije pojavio.)
"""



"""
broj	puta
9
22
25
16
30
14
19
14
36
13
15
13
29
12
16
12
21
11
10
11
14
10
13
10
31
9
8
9
35
8
24
8
34
7
32
7
3
7
2
7
27
6
26
6
20
5
17
4
7
4
5
4
33
3
23
3
22
3
1
3
37
2
12
2
38
1


0×: 39, 28, 18, 11, 6, 4



Najmanje (1×): 38

39. → [4, 6, 11, 18, 28, 38, 39]
"""
