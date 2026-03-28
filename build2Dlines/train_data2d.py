
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone 2D training data generator (CPython).

Inputs:
- Seismic volume: float32 little-endian .dat, shape (n3, n2, n1)
- Impedance volume: float32 little-endian .dat, shape (n3, n2, n1)
- Wells: txt files with columns: iline xline twt value
- Metadata JSON: metadata.json with keys:
  n1,d1,f1,n_inline,n_xline,d_inline,d_xline,inline0,xline0

Outputs:
- out_dir/<split>/sx/*.dat : seismic 2D patches (nPath, n1)
- out_dir/<split>/ws/*.dat : smoothed impedance 2D patches (nPath, n1)
- out_dir/<split>/wx/*.dat : sparse well-log 2D patches (nPath, n1) with 0 elsewhere

Key behaviors:
1) Scan well_dir automatically (no need to pass each well file)
2) valid_wells is EXACT match on file stems (wf.stem). No prefix match.
3) Train excludes validation wells; valid samples MUST pass at least one valid well.
4) If pool has only 1 well, synthesize a 2nd control point so a path can still be built.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set

import numpy as np


# -------------------------
# Meta & data structures
# -------------------------
@dataclass(frozen=True)
class GridMeta:
    n1: int
    n2: int
    n3: int
    d1: float
    d2: float
    d3: float
    f1: float
    f2: float
    f3: float


@dataclass(frozen=True)
class WellCenter:
    name: str
    k3: int
    k2: int


def get_io_config() -> dict:
    base = Path(r"../")
    return {
        "metadata": base / "out" / "metadata.json",
        "seis": base / "out" / "seis" / "seis.dat",
        "imp": base / "out" / "seis" / "initial_model.dat",

        # ✅ well folder scanning
        "well_dir": base / "out" / "well",
        "well_pattern": "*.txt",
        "well_recursive": False,

        "out_dir": base / "out" / "seis" / "train",

        "train_samples": 100,
        "valid_samples": 50,
        "train_subdir": "trainimp",
        "valid_subdir": "validimp",

        # path control points
        "path_points_min": 2,
        "path_points_max": 2,
        "path_dist_min": 2.0,
        "path_dist_max": 2000.0,
        "path_cos_straight_reject": 0.5,
        "path_max_tries": 20000,

        # split control
        "split_by_well": True,
        "valid_well_fraction": 0.5,  # only used when valid_wells not specified

        # smoothing
        "sigma_xline": 6.0,
        "sigma_twt": 1.0,

        "seed": 20260108,

        # well name of validation wells：
        "valid_wells": ["w1"],  #  ["Well1"]
    }


# -------------------------
# Utils
# -------------------------
def scan_well_files(well_dir: Path, pattern: str = "*.txt", recursive: bool = False) -> List[Path]:
    if not well_dir.exists():
        raise FileNotFoundError(f"well_dir not found: {well_dir}")
    files = list(well_dir.rglob(pattern) if recursive else well_dir.glob(pattern))
    files = [f for f in files if f.is_file()]
    return sorted(files)


def load_metadata(path: Path) -> GridMeta:
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    n1 = int(meta["n1"])
    d1 = float(meta["d1"])
    f1 = float(meta["f1"])

    n3 = int(meta["n_inline"])
    n2 = int(meta["n_xline"])

    d3 = float(meta["d_inline"])
    d2 = float(meta["d_xline"])

    f3 = float(meta["inline0"])
    f2 = float(meta["xline0"])

    return GridMeta(n1=n1, n2=n2, n3=n3, d1=d1, d2=d2, d3=d3, f1=f1, f2=f2, f3=f3)


def open_volume_memmap(path: Path, shape: Tuple[int, int, int]) -> np.memmap:
    return np.memmap(str(path), dtype="<f4", mode="r", shape=shape, order="C")


def gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    if radius is None:
        radius = int(math.ceil(3.0 * sigma))
    radius = max(1, radius)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def convolve1d_same(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.convolve(a, k, mode="same")


def gaussian_smooth_2d(slice2d: np.ndarray, sigma_xline: float, sigma_twt: float) -> np.ndarray:
    out = slice2d.astype(np.float64, copy=False)
    kx = gaussian_kernel1d(sigma_xline)
    kt = gaussian_kernel1d(sigma_twt)

    if kx.size > 1:
        tmp = np.empty_like(out, dtype=np.float64)
        for j in range(out.shape[1]):
            tmp[:, j] = convolve1d_same(out[:, j], kx)
        out = tmp

    if kt.size > 1:
        tmp = np.empty_like(out, dtype=np.float64)
        for i in range(out.shape[0]):
            tmp[i, :] = convolve1d_same(out[i, :], kt)
        out = tmp

    return out.astype(np.float32)


def _to_index(v: float, f: float, d: float) -> int:
    return int(round((v - f) / d))


def _to_coord(k: int, f: float, d: float) -> float:
    return float(f + k * d)


# -------------------------
# Well reading & sparse index
# -------------------------
def read_well_file(
    path: Path, meta: GridMeta
) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int, int, float]]]:
    """
    Read one well txt file (iline xline twt value) -> (k3,k2,k1,val).

    NOTE: no resampling. Only map twt to nearest k1 by metadata f1/d1.
    Ensure your twt unit matches metadata (usually seconds).
    """
    if not path.exists():
        return None, []

    samples: List[Tuple[int, int, int, float]] = []
    k3_list: List[int] = []
    k2_list: List[int] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                iline = float(parts[0])
                xline = float(parts[1])
                twt = float(parts[2])
                val = float(parts[3])
            except Exception:
                continue

            k3 = _to_index(iline, meta.f3, meta.d3)
            k2 = _to_index(xline, meta.f2, meta.d2)
            k1 = _to_index(twt, meta.f1, meta.d1)

            if not (0 <= k3 < meta.n3 and 0 <= k2 < meta.n2 and 0 <= k1 < meta.n1):
                continue

            samples.append((int(k3), int(k2), int(k1), float(val)))
            k3_list.append(int(k3))
            k2_list.append(int(k2))

    if not samples:
        return None, []

    # ✅ center uses median (more stable than "first row")
    c3 = int(np.round(np.median(np.asarray(k3_list, dtype=np.float64))))
    c2 = int(np.round(np.median(np.asarray(k2_list, dtype=np.float64))))
    return (c3, c2), samples


def build_well_sparse_index(
    well_files: List[Path], meta: GridMeta
) -> Tuple[List[WellCenter], Dict[int, Dict[int, Dict[int, float]]]]:
    centers: List[WellCenter] = []
    by_inline: Dict[int, Dict[int, Dict[int, float]]] = {}

    bad = 0
    for wf in well_files:
        first_pos, samples = read_well_file(wf, meta)
        if first_pos is not None:
            k3, k2 = first_pos
            centers.append(WellCenter(name=wf.stem, k3=int(k3), k2=int(k2)))
        else:
            bad += 1

        for k3, k2, k1, val in samples:
            by_inline.setdefault(int(k3), {}).setdefault(int(k2), {})[int(k1)] = float(val)

    if bad > 0:
        print(f"[WARN] {bad} well files have no valid samples inside volume (not used as centers).")

    return centers, by_inline


def _split_positions_by_well(
    positions: List[Tuple[int, int]], rng: random.Random, valid_fraction: float
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    if not positions:
        return [], []
    if len(positions) == 1:
        return positions[:], positions[:]
    valid_fraction = max(0.0, min(1.0, valid_fraction))
    n_valid = int(round(valid_fraction * len(positions)))
    n_valid = max(1, min(len(positions) - 1, n_valid))
    idxs = list(range(len(positions)))
    rng.shuffle(idxs)
    valid_idxs = set(idxs[:n_valid])
    train_pos = [p for i, p in enumerate(positions) if i not in valid_idxs]
    valid_pos = [p for i, p in enumerate(positions) if i in valid_idxs]
    return train_pos, valid_pos


# -------------------------
# Path construction
# -------------------------
def _check_next_point(
    k2s: List[int],
    k3s: List[int],
    i2: int,
    i3: int,
    dist_min: float,
    dist_max: float,
    cos_straight_reject: float,
) -> bool:
    """Return True if candidate should be rejected."""
    if not k2s:
        return False

    dp2 = float(i2 - k2s[-1])
    dp3 = float(i3 - k3s[-1])
    dps = math.sqrt(dp2 * dp2 + dp3 * dp3)
    if dps < dist_min or dps > dist_max:
        return True

    if len(k2s) >= 2:
        ds2 = float(k2s[-1] - k2s[-2])
        ds3 = float(k3s[-1] - k3s[-2])
        ds = math.sqrt(ds2 * ds2 + ds3 * ds3)
        if ds > 0.0 and dps > 0.0:
            d2s = (ds2 * dp2 + ds3 * dp3) / (ds * dps)
            if d2s > cos_straight_reject:
                return True
    return False


def _synthesize_second_point(
    rng: random.Random,
    meta: GridMeta,
    k2s: List[int],
    k3s: List[int],
    dist_min: float,
    dist_max: float,
    cos_straight_reject: float,
) -> bool:
    """If only 1 control point exists, create a second point within [dist_min, dist_max]."""
    if len(k2s) != 1 or len(k3s) != 1:
        return False

    i2c, i3c = int(k2s[0]), int(k3s[0])
    for _ in range(2000):
        ang = rng.random() * 2.0 * math.pi
        L = rng.uniform(float(dist_min), float(dist_max))
        i2 = int(round(i2c + L * math.cos(ang)))
        i3 = int(round(i3c + L * math.sin(ang)))

        i2 = int(max(0, min(meta.n2 - 1, i2)))
        i3 = int(max(0, min(meta.n3 - 1, i3)))
        if i2 == i2c and i3 == i3c:
            continue

        if _check_next_point(k2s, k3s, i2, i3, dist_min, dist_max, cos_straight_reject):
            continue

        k2s.append(i2)
        k3s.append(i3)
        return True
    return False


def _rand_near_boundary_point(meta, rng, strip: int = 16, margin: int = 2, side: Optional[int] = None):
    """
    在工区边界附近(strip宽度)随机取一个点 (k2,k3)。
    side: None=随机四边; 0=left,1=right,2=bottom,3=top
    """
    n2, n3 = int(meta.n2), int(meta.n3)
    strip = max(1, int(strip))
    margin = max(0, int(margin))

    # 防止 strip 过大
    strip2 = min(strip, max(1, (n2 - 2*margin)//2))
    strip3 = min(strip, max(1, (n3 - 2*margin)//2))

    if side is None:
        side = rng.randrange(4)

    if side == 0:  # left
        k2 = rng.randint(margin, margin + strip2 - 1)
        k3 = rng.randint(margin, n3 - 1 - margin)
    elif side == 1:  # right
        k2 = rng.randint(n2 - strip2 - margin, n2 - 1 - margin)
        k3 = rng.randint(margin, n3 - 1 - margin)
    elif side == 2:  # bottom
        k3 = rng.randint(margin, margin + strip3 - 1)
        k2 = rng.randint(margin, n2 - 1 - margin)
    else:  # top
        k3 = rng.randint(n3 - strip3 - margin, n3 - 1 - margin)
        k2 = rng.randint(margin, n2 - 1 - margin)

    k2 = int(max(0, min(n2 - 1, k2)))
    k3 = int(max(0, min(n3 - 1, k3)))
    return k2, k3, int(side)


def _opposite_side(side: int) -> int:
    # 0<->1, 2<->3
    return {0: 1, 1: 0, 2: 3, 3: 2}.get(int(side), 1)

def get_random_path(
    centers: Sequence[WellCenter],
    rng: random.Random,
    meta: GridMeta,
    validation_wells: Optional[set[str]],
    path_points_min: int,
    path_points_max: int,
    dist_min: float,
    dist_max: float,
    cos_straight_reject: float,
    max_tries: int,
    must_include_wells: Optional[Set[str]] = None,
    boundary_strip: int = 16,   # 
    boundary_margin: int = 2,   # 
) -> Optional[Tuple[List[float], List[float]]]:

    if not centers:
        return None

    validation_wells = validation_wells or set()
    must_include_wells = must_include_wells or set()

    # 
    pool = [c for c in centers if c.name not in validation_wells]
    if len(pool) == 0:
        return None

    # ----------  ----------
    # 
    if len(pool) == 1:
        w = pool[0]
        w2, w3 = int(w.k2), int(w.k3)

        for _ in range(5000):
            b2, b3, bside = _rand_near_boundary_point(meta, rng, strip=boundary_strip, margin=boundary_margin, side=None)
            eside = _opposite_side(bside)
            e2, e3, _ = _rand_near_boundary_point(meta, rng, strip=boundary_strip, margin=boundary_margin, side=eside)

            # 
            if (b2 == w2 and b3 == w3) or (e2 == w2 and e3 == w3):
                continue
            if (b2 == e2 and b3 == e3):
                continue

            # 
            db = math.sqrt((b2 - w2)**2 + (b3 - w3)**2)
            de = math.sqrt((e2 - w2)**2 + (e3 - w3)**2)
            if db < dist_min or de < dist_min:
                continue

            # 
            k2e = [int(b2), int(w2), int(e2)]
            k3e = [int(b3), int(w3), int(e3)]

            p2 = [_to_coord(k, meta.f2, meta.d2) for k in k2e]
            p3 = [_to_coord(k, meta.f3, meta.d3) for k in k3e]
            return p2, p3

        return None


    pool_map = {c.name: c for c in pool}
    missing = [n for n in must_include_wells if n not in pool_map]
    if missing:
        return None

    must_list = [pool_map[n] for n in must_include_wells]

    np_target = rng.randint(int(path_points_min), int(path_points_max))
    np_target = max(2, min(len(pool), np_target))
    np_target = max(np_target, len(must_list))

    for _ in range(max_tries):
        k2s: List[int] = []
        k3s: List[int] = []
        used_names: Set[str] = set()

   
        order = list(must_list)
        rng.shuffle(order)
        ok = True
        for c in order:
            i2, i3 = int(c.k2), int(c.k3)
            if _check_next_point(k2s, k3s, i2, i3, dist_min, dist_max, cos_straight_reject):
                ok = False
                break
            k2s.append(i2); k3s.append(i3)
            used_names.add(c.name)
        if not ok:
            continue


        tries_inner = 0
        while len(k2s) < np_target and tries_inner < max_tries:
            tries_inner += 1
            c = rng.choice(pool)
            if c.name in used_names:
                continue
            i2, i3 = int(c.k2), int(c.k3)
            if _check_next_point(k2s, k3s, i2, i3, dist_min, dist_max, cos_straight_reject):
                continue
            k2s.append(i2); k3s.append(i3)
            used_names.add(c.name)

        if len(k2s) < 2:
            continue


        b2 = 2 * k2s[0] - k2s[1]
        b3 = 2 * k3s[0] - k3s[1]
        e2 = 2 * k2s[-1] - k2s[-2]
        e3 = 2 * k3s[-1] - k3s[-2]

        b2 = int(max(0, min(meta.n2 - 1, b2)))
        e2 = int(max(0, min(meta.n2 - 1, e2)))
        b3 = int(max(0, min(meta.n3 - 1, b3)))
        e3 = int(max(0, min(meta.n3 - 1, e3)))

        k2e = [b2] + k2s + [e2]
        k3e = [b3] + k3s + [e3]

        p2 = [_to_coord(k, meta.f2, meta.d2) for k in k2e]
        p3 = [_to_coord(k, meta.f3, meta.d3) for k in k3e]
        return p2, p3

    return None



# -------------------------
# Extraction
# -------------------------
def _extract_trace_from_sparse(
    sparse_by_inline: Dict[int, Dict[int, Dict[int, float]]],
    k3: int,
    k2: int,
    n1: int,
) -> Tuple[np.ndarray, int]:
    """
    Return a sparse well-log trace along time/depth axis (length n1).

    - Where no well-log sample exists, fill with 0 (NOT NaN).
    - Also return hit_count (# of samples written), so caller can still reject
      paths that contain no well information at all.
    """
    out = np.zeros((n1,), dtype=np.float32)
    hit = 0
    by_k2 = sparse_by_inline.get(int(k3), {})
    if not by_k2:
        return out, hit
    by_k1 = by_k2.get(int(k2), {})
    if not by_k1:
        return out, hit
    for kk1, val in by_k1.items():
        k1i = int(kk1)
        if 0 <= k1i < n1:
            out[k1i] = np.float32(val)
            hit += 1
    return out, hit



def random_extraction(
    rng: random.Random,
    meta: GridMeta,
    sx_vol: np.memmap,
    wi_vol: np.memmap,
    ws_cache: Dict[int, np.ndarray],
    wells_sparse_by_inline: Dict[int, Dict[int, Dict[int, float]]],
    centers: Sequence[WellCenter],
    exclude_wells: Set[str],
    sigma_xline: float,
    sigma_twt: float,
    path_points_min: int,
    path_points_max: int,
    path_dist_min: float,
    path_dist_max: float,
    path_cos_straight_reject: float,
    path_max_tries: int,
    must_include_wells: Optional[Set[str]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    ps = get_random_path(
        centers=centers,
        rng=rng,
        meta=meta,
        validation_wells=exclude_wells,
        must_include_wells=must_include_wells,
        path_points_min=path_points_min,
        path_points_max=path_points_max,
        dist_min=path_dist_min,
        dist_max=path_dist_max,
        cos_straight_reject=path_cos_straight_reject,
        max_tries=path_max_tries,
    )
    if ps is None:
        return None

    p2, p3 = ps
    if len(p2) < 2:
        return None
    sx_traces: List[np.ndarray] = []
    wx_traces: List[np.ndarray] = []
    ws_traces: List[np.ndarray] = []

    total_hits = 0

    # Deduplicate consecutive (j2,j3) samples to avoid double-counting at segment junctions
    # (e.g., single-well paths [start, well, end] include the well point in both segments).
    last_j2: Optional[int] = None
    last_j3: Optional[int] = None

    for ip in range(len(p2) - 1):
        b2c, b3c = float(p2[ip]), float(p3[ip])
        e2c, e3c = float(p2[ip + 1]), float(p3[ip + 1])

        b2 = _to_index(b2c, meta.f2, meta.d2)
        b3 = _to_index(b3c, meta.f3, meta.d3)
        e2 = _to_index(e2c, meta.f2, meta.d2)
        e3 = _to_index(e3c, meta.f3, meta.d3)

        d2 = float(e2 - b2)
        d3 = float(e3 - b3)
        ds = math.sqrt(d2 * d2 + d3 * d3)
        nstep = int(max(1, math.floor(ds)))

        # ✅ include endpoint so control points are sampled
        for ik in range(nstep + 1):
            x2 = float(b2) + (ik * d2 / ds) if ds > 0 else float(b2)
            x3 = float(b3) + (ik * d3 / ds) if ds > 0 else float(b3)
            j2 = int(round(x2))
            j3 = int(round(x3))
            if not (0 <= j2 < meta.n2 and 0 <= j3 < meta.n3):
                continue

            # skip consecutive duplicates (can happen due to rounding or shared control points)
            if last_j2 == j2 and last_j3 == j3:
                continue
            last_j2, last_j3 = j2, j3

            sx_trace = np.asarray(sx_vol[j3, j2, :], dtype=np.float32)

            if j3 not in ws_cache:
                wi_slice = np.asarray(wi_vol[j3, :, :], dtype=np.float32)
                ws_cache[j3] = gaussian_smooth_2d(wi_slice, sigma_xline=sigma_xline, sigma_twt=sigma_twt)
            ws_trace = np.asarray(ws_cache[j3][j2, :], dtype=np.float32)

            wx_trace, nhit = _extract_trace_from_sparse(wells_sparse_by_inline, j3, j2, meta.n1)
            total_hits += nhit

            sx_traces.append(sx_trace)
            wx_traces.append(wx_trace)
            ws_traces.append(ws_trace)

    if not sx_traces:
        return None

    se = np.stack(sx_traces, axis=0)
    ve = np.stack(wx_traces, axis=0)
    pe = np.stack(ws_traces, axis=0)

    # If the whole path never touched any well samples, reject (keep dataset meaningful).
    if total_hits <= 0:
        return None

    return se, ve, pe


def write_patch(path: Path, patch: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    patch.astype("<f4", copy=False).tofile(str(path))


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone 2D training data generator")
    p.add_argument("--metadata", type=str, default=None)
    p.add_argument("--seis", type=str, default=None)
    p.add_argument("--imp", type=str, default=None)

    p.add_argument("--well_dir", type=str, default=None, help="folder containing well txt files")
    p.add_argument("--well_pattern", type=str, default=None, help='glob pattern, e.g. "*.txt"')
    p.add_argument("--well_recursive", type=int, default=None, help="1=True, 0=False")

    # optional fallback
    p.add_argument("--wells", type=str, nargs="*", default=None)

    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--train_samples", type=int, default=None)
    p.add_argument("--valid_samples", type=int, default=None)
    p.add_argument("--train_subdir", type=str, default=None)
    p.add_argument("--valid_subdir", type=str, default=None)

    p.add_argument("--split_by_well", type=int, default=None, help="1=True, 0=False")
    p.add_argument("--valid_well_fraction", type=float, default=None)

    p.add_argument("--path_points_min", type=int, default=None)
    p.add_argument("--path_points_max", type=int, default=None)
    p.add_argument("--path_dist_min", type=float, default=None)
    p.add_argument("--path_dist_max", type=float, default=None)
    p.add_argument("--path_cos_straight_reject", type=float, default=None)
    p.add_argument("--path_max_tries", type=int, default=None)

    p.add_argument("--sigma_xline", type=float, default=None)
    p.add_argument("--sigma_twt", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)

    # ✅ user-specified validation wells (EXACT match on wf.stem)
    p.add_argument("--valid_wells", type=str, nargs="*", default=None, help="validation well names (exact wf.stem)")

    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main() -> int:
    args = parse_args()
    cfg = get_io_config()

    metadata = Path(args.metadata) if args.metadata else Path(cfg["metadata"])
    seis = Path(args.seis) if args.seis else Path(cfg["seis"])
    imp = Path(args.imp) if args.imp else Path(cfg["imp"])
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["out_dir"])

    train_samples = int(args.train_samples) if args.train_samples is not None else int(cfg["train_samples"])
    valid_samples = int(args.valid_samples) if args.valid_samples is not None else int(cfg["valid_samples"])
    train_subdir = str(args.train_subdir) if args.train_subdir is not None else str(cfg["train_subdir"])
    valid_subdir = str(args.valid_subdir) if args.valid_subdir is not None else str(cfg["valid_subdir"])

    if args.split_by_well is None:
        split_by_well = bool(cfg.get("split_by_well", True))
    else:
        split_by_well = bool(int(args.split_by_well))

    valid_well_fraction = (
        float(args.valid_well_fraction)
        if args.valid_well_fraction is not None
        else float(cfg.get("valid_well_fraction", 0.5))
    )

    path_points_min = int(args.path_points_min) if args.path_points_min is not None else int(cfg["path_points_min"])
    path_points_max = int(args.path_points_max) if args.path_points_max is not None else int(cfg["path_points_max"])
    path_dist_min = float(args.path_dist_min) if args.path_dist_min is not None else float(cfg["path_dist_min"])
    path_dist_max = float(args.path_dist_max) if args.path_dist_max is not None else float(cfg["path_dist_max"])
    path_cos_straight_reject = (
        float(args.path_cos_straight_reject)
        if args.path_cos_straight_reject is not None
        else float(cfg["path_cos_straight_reject"])
    )
    path_max_tries = int(args.path_max_tries) if args.path_max_tries is not None else int(cfg["path_max_tries"])
    sigma_xline = float(args.sigma_xline) if args.sigma_xline is not None else float(cfg["sigma_xline"])
    sigma_twt = float(args.sigma_twt) if args.sigma_twt is not None else float(cfg["sigma_twt"])
    seed = int(args.seed) if args.seed is not None else int(cfg["seed"])

    rng = random.Random(seed)
    np.random.seed(seed)

    if not metadata.exists():
        raise FileNotFoundError(f"metadata not found: {metadata}")
    if not seis.exists():
        raise FileNotFoundError(f"seis not found: {seis}")
    if not imp.exists():
        raise FileNotFoundError(f"imp not found: {imp}")

    meta = load_metadata(metadata)
    shape = (meta.n3, meta.n2, meta.n1)
    print(f"[INFO] Grid shape (n3,n2,n1) = {shape}")

    print(f"[INFO] Reading seismic memmap: {seis}")
    sx_vol = open_volume_memmap(seis, shape)
    print(f"[INFO] Reading impedance memmap: {imp}")
    wi_vol = open_volume_memmap(imp, shape)

    # ---- scan wells
    well_dir = Path(args.well_dir) if args.well_dir else Path(cfg.get("well_dir"))
    well_pattern = str(args.well_pattern) if args.well_pattern is not None else str(cfg.get("well_pattern", "*.txt"))
    well_recursive = bool(int(args.well_recursive)) if args.well_recursive is not None else bool(cfg.get("well_recursive", False))

    wells: List[Path] = []
    if well_dir is not None and str(well_dir) != "None":
        wells = scan_well_files(well_dir, pattern=well_pattern, recursive=well_recursive)

    # fallback: explicit list
    if not wells:
        if args.wells:
            wells = [Path(w) for w in args.wells]
        else:
            wells = [Path(w) for w in cfg.get("wells", [])]

    if not wells:
        raise RuntimeError("No well files found. Provide --well_dir (recommended) or --wells.")

    print(f"[INFO] Found {len(wells)} well files (show first 10):")
    for wp in wells[:10]:
        print(f"       - {wp}")
    if len(wells) > 10:
        print(f"       ... ({len(wells)-10} more)")

    # ---- build sparse index
    print("[INFO] Building sparse well index...")
    centers, wells_by_inline = build_well_sparse_index(wells, meta)
    if not centers:
        raise RuntimeError("No valid well positions parsed. Check WELL txt files and metadata.")
    print(f"[INFO] Parsed {len(centers)} well centers, sparse points in {len(wells_by_inline)} inline slices")

    center_names = {c.name for c in centers}

    # ---- resolve user valid wells (EXACT)
    user_valid_set: Set[str] = set()
    if args.valid_wells is not None and len(args.valid_wells) > 0:
        user_valid_set = {s.strip() for s in args.valid_wells if s.strip()}
    else:
        user_valid_set = {str(s).strip() for s in cfg.get("valid_wells", []) if str(s).strip()}

    if user_valid_set:
        missing = sorted(list(user_valid_set - center_names))
        if missing:
            raise ValueError(
                f"[ERROR] valid_wells contains names not found (must equal well file stems): {missing}\n"
                f"        Available stems (centers): {sorted(list(center_names))}"
            )

    # ---- split train/valid
    if split_by_well:
        if user_valid_set:
            valid_centers = [c for c in centers if c.name in user_valid_set]
            train_centers = [c for c in centers if c.name not in user_valid_set]
            validation_wells_names = {c.name for c in valid_centers}
        else:
            # fallback fraction split
            positions = [(c.k3, c.k2) for c in centers]
            train_pos, valid_pos = _split_positions_by_well(positions, rng, valid_well_fraction)
            train_set = set(train_pos)
            valid_set = set(valid_pos)
            train_centers = [c for c in centers if (c.k3, c.k2) in train_set]
            valid_centers = [c for c in centers if (c.k3, c.k2) in valid_set]
            validation_wells_names = {c.name for c in valid_centers}
    else:
        train_centers = list(centers)
        valid_centers = list(centers)
        validation_wells_names = set()

    print(f"[INFO] split_by_well={split_by_well} train_wells={len(train_centers)} valid_wells={len(valid_centers)}")
    if user_valid_set:
        print(f"[INFO] user valid_wells(exact)={sorted(list(user_valid_set))}")
        print(f"[INFO] matched valid centers={[c.name for c in valid_centers]}")

    # ---- output dirs
    out_train_ws = out_dir / train_subdir / "ws"
    out_train_wx = out_dir / train_subdir / "wx"
    out_train_sx = out_dir / train_subdir / "sx"
    out_valid_ws = out_dir / valid_subdir / "ws"
    out_valid_wx = out_dir / valid_subdir / "wx"
    out_valid_sx = out_dir / valid_subdir / "sx"
    for d in (out_train_ws, out_train_wx, out_train_sx, out_valid_ws, out_valid_wx, out_valid_sx):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output dir: {out_dir} (layout: {train_subdir}/..., {valid_subdir}/...)")

    ws_cache: Dict[int, np.ndarray] = {}

    def _generate_like_jython(
        split_name: str,
        loops: int,
        centers_for_path: Sequence[WellCenter],
        out_sx: Path,
        out_wx: Path,
        out_ws: Path,
        kd0: int,
        exclude_wells: Set[str],
        must_include_wells: Optional[Set[str]],
    ) -> int:
        kd = int(kd0)
        wrote = 0
        for _ in range(int(loops)):
            re = random_extraction(
                rng=rng,
                meta=meta,
                sx_vol=sx_vol,
                wi_vol=wi_vol,
                ws_cache=ws_cache,
                wells_sparse_by_inline=wells_by_inline,
                centers=centers_for_path,
                exclude_wells=exclude_wells,
                sigma_xline=sigma_xline,
                sigma_twt=sigma_twt,
                path_points_min=path_points_min,
                path_points_max=path_points_max,
                path_dist_min=path_dist_min,
                path_dist_max=path_dist_max,
                path_cos_straight_reject=path_cos_straight_reject,
                path_max_tries=path_max_tries,
                must_include_wells=must_include_wells,
            )
            if re is None:
                continue
            se, ve, pe = re
            npath = int(pe.shape[0])
            name = f"{kd}-{npath}.dat"
            write_patch(out_ws / name, pe)
            write_patch(out_wx / name, ve)
            write_patch(out_sx / name, se)
            kd += 1
            wrote += 1
            if wrote % 20 == 0:
                print(f"[INFO] {split_name}: wrote {wrote}/{loops} (kd={kd})")
        if wrote < loops:
            print(f"[WARN] {split_name}: wrote {wrote}/{loops} (some extractions rejected)")
        return kd

    kd = 0
    print("[INFO] Generating train samples...")
    kd = _generate_like_jython(
        split_name="train",
        loops=train_samples,
        centers_for_path=train_centers if split_by_well else centers,
        out_sx=out_train_sx,
        out_wx=out_train_wx,
        out_ws=out_train_ws,
        kd0=kd,
        exclude_wells=validation_wells_names,  # ✅ train excludes valid wells
        must_include_wells=None,
    )

    print("[INFO] Generating valid samples...")
    kd = _generate_like_jython(
        split_name="valid",
        loops=valid_samples,
        centers_for_path=valid_centers if split_by_well else centers,
        out_sx=out_valid_sx,
        out_wx=out_valid_wx,
        out_ws=out_valid_ws,
        kd0=kd,
        exclude_wells=set(),
        must_include_wells=user_valid_set if user_valid_set else None,  # ✅ must pass at least one of them
    )

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
