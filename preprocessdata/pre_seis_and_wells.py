#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
from segysak.segy import segy_loader

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xarray as xr




SEGY_SEIS_PATH = r"../oridata/seis/seis.segy"
SEGY_UF_PATH = r"../oridata/seis/rgt.segy"



OUT_SEIS_DAT = r"../out/seis/seis.dat"
OUT_UF_DAT = r"../out/seis/rgt.dat"
OUT_METADATA = r"../out/metadata.json"


#    .xlsx / .xls / .csv / .txt
WELL_FILES = [
    r"/path/to/well1.xlsx",
    # r"/path/to/well2.xlsx",
]


WELL_DIR = None  
WELL_DIR = r"../oridata/well"


OUT_WELL_DIR = r"../out/well/"

# 6) Vertical first sample f1 (same unit as TWT, usually 0.0)
#    If None, we use the first vertical coordinate from SEGY (e.g., 0.0 ms)
F1_OVERRIDE = None  # e.g., 0.0

# 7) Header byte positions (VERY IMPORTANT)
#    These are "byte positions" in the trace header used by segysak/xarray:
#      inline  : ILINE_BYTE
#      xline   : XLINE_BYTE
#      CDP X   : CDP_X_BYTE
#      CDP Y   : CDP_Y_BYTE
#    Example values: inline=189, xline=193, cdp_x=73, cdp_y=77
ILINE_BYTE = 5
XLINE_BYTE = 21
CDP_X_BYTE = 73
CDP_Y_BYTE = 77

# 8) Vertical dimension name hint
#    Usually segysak/xarray will give one of: "twt", "time", "samples", "depth".
#    If None, script will auto-detect from these candidates.
VERT_DIM_HINT = "TWT"

# =============== CONFIG 结束 ================================


# ---------- 地震处理部分 -------------------------------------

def load_seismic_with_segysak(segy_path: str):
    """

    """
    segy_path = Path(segy_path)
    if not segy_path.exists():
        raise FileNotFoundError(f"Seismic SEGY not found: {segy_path}")

    print(f"\n[SEGY] Loading seismic with segy_loader: {segy_path}")

    #
    ds = segy_loader(
        str(segy_path),
        iline=ILINE_BYTE,
        xline=XLINE_BYTE,
        cdpx=CDP_X_BYTE,
        cdpy=CDP_Y_BYTE,
        # 
    )

    print("[SEGY] dims      :", ds.dims)
    print("[SEGY] coords    :", list(ds.coords))
    print("[SEGY] data_vars :", list(ds.data_vars))

    #
    if VERT_DIM_HINT is not None and VERT_DIM_HINT in ds.dims:
        vert_dim = VERT_DIM_HINT
    else:
        for cand in ("twt", "time", "samples", "depth"):
            if cand in ds.dims:
                vert_dim = cand
                break
        else:
            raise RuntimeError(
                f"Could not find vertical dimension; ds.dims = {ds.dims}. "
                f"Specify VERT_DIM_HINT explicitly."
            )

    tcoord = ds[vert_dim].values.astype(float)
    n1 = tcoord.size
    if n1 < 2:
        raise RuntimeError("Vertical coordinate has <2 samples; check SEGY vertical axis.")

    d1 = float(tcoord[1] - tcoord[0])
    if F1_OVERRIDE is not None:
        f1 = float(F1_OVERRIDE)
    else:
        f1 = float(tcoord[0])

    # inline / xline 
    iline_coord = ds["iline"].values.astype(int)
    xline_coord = ds["xline"].values.astype(int)
    n_inline = iline_coord.size
    n_xline = xline_coord.size

    print(f"[SEGY] inline range: {iline_coord[0]} -> {iline_coord[-1]} (n={n_inline})")
    print(f"[SEGY] xline  range: {xline_coord[0]} -> {xline_coord[-1]} (n={n_xline})")
    print(f"[SEGY] n1={n1}, d1={d1}, f1={f1}, vert_dim={vert_dim}")

    # 
    if "data" in ds.data_vars:
        data_var = "data"
    else:
        data_var = list(ds.data_vars)[0]
        print(f"[SEGY] WARNING: no variable named 'data', using '{data_var}' as volume.")

    # 
    vol_da = ds[data_var].transpose("iline", "xline", vert_dim)
    vol = vol_da.values.astype(np.float32)
    print("[SEGY] volume shape =", vol.shape)

    # 
    if ("cdp_x" not in ds.variables) or ("cdp_y" not in ds.variables):
        raise RuntimeError(
            "Dataset has no 'cdp_x' or 'cdp_y'. "
            ""
        )

    xg = ds["cdp_x"].values.astype(float)
    yg = ds["cdp_y"].values.astype(float)

    if xg.shape != (n_inline, n_xline):
        print("[SEGY] WARNING: cdp_x shape", xg.shape, "!= (n_inline, n_xline)", (n_inline, n_xline))
    if yg.shape != (n_inline, n_xline):
        print("[SEGY] WARNING: cdp_y shape", yg.shape, "!= (n_inline, n_xline)", (n_inline, n_xline))

    # 
    points_xy = np.column_stack([xg.ravel(), yg.ravel()])

    # 
    il_grid, xl_grid = np.meshgrid(iline_coord, xline_coord, indexing="ij")
    iline_flat = il_grid.ravel().astype(int)
    xline_flat = xl_grid.ravel().astype(int)

    # metadata
    inline0 = float(iline_coord[0])
    xline0 = float(xline_coord[0])
    d_inline = float(iline_coord[1] - inline0) if n_inline > 1 else 1.0
    d_xline = float(xline_coord[1] - xline0)  if n_xline > 1 else 1.0

    meta = dict(
        n1=int(n1),
        d1=float(d1),
        f1=float(f1),
        n_inline=int(n_inline),
        n_xline=int(n_xline),
        inline0=float(inline0),
        xline0=float(xline0),
        d_inline=float(d_inline),
        d_xline=float(d_xline),
        iline_vals=iline_coord.tolist(),
        xline_vals=xline_coord.tolist(),
        vert_dim=str(vert_dim),
        order="iline_xline_time",
    )

    return vol, meta, points_xy, iline_flat, xline_flat


def export_dat(volume: np.ndarray, out_path: str, name: str):
    """Save 3D volume to float32 .dat file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    volume.astype(np.float32).tofile(out_path)
    print(f"[WRITE] {name}.dat -> {out_path} (shape={volume.shape})")


def export_metadata(meta: dict, out_path: str):
    """Save metadata.json."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(meta, fw, indent=2)
    print(f"[WRITE] metadata.json -> {out_path}")


def load_uf_with_segysak(segy_path: str, meta: dict):
    """
    用 segy_loader 读取 RGT (uf) 体，沿用 seismic 的 iline/xline/vert 规则。
    """
    if segy_path is None:
        print("[UF] SEGY_UF_PATH is None, skip uf.")
        return None

    segy_path = Path(segy_path)
    if not segy_path.exists():
        print(f"[UF] {segy_path} not found, skip uf.")
        return None

    print(f"\n[UF] Loading uf SEGY with segy_loader: {segy_path}")

    ds = segy_loader(
        str(segy_path),
        iline=ILINE_BYTE,
        xline=XLINE_BYTE,
        # 
    )

    print("[UF] dims      :", ds.dims)
    print("[UF] coords    :", list(ds.coords))
    print("[UF] data_vars :", list(ds.data_vars))

    # 
    vert_dim_meta = meta.get("vert_dim", None)
    if vert_dim_meta is not None and vert_dim_meta in ds.dims:
        vert_dim = vert_dim_meta
    else:
        for cand in ("twt", "time", "samples", "depth"):
            if cand in ds.dims:
                vert_dim = cand
                break
        else:
            raise RuntimeError(f"[UF] Could not find vertical dim; ds.dims={ds.dims}")

    if "data" in ds.data_vars:
        data_var = "data"
    else:
        data_var = list(ds.data_vars)[0]
        print(f"[UF] WARNING: no 'data' variable; using '{data_var}' as uf.")

    uf_da = ds[data_var].transpose("iline", "xline", vert_dim)
    uf = uf_da.values.astype(np.float32)
    print("[UF] volume shape =", uf.shape)

    n_inline = int(meta["n_inline"])
    n_xline  = int(meta["n_xline"])
    n1       = int(meta["n1"])
    if uf.shape != (n_inline, n_xline, n1):
        print(f"[UF] WARNING: uf shape {uf.shape} != seismic ({n_inline},{n_xline},{n1}).")

    return uf


def load_well_table_with_name(path: str):
    """

    """
    p = Path(path)
    raw = pd.read_excel(p, header=None, dtype=str)

    # 
    first_row = raw.iloc[0]
    wellname = None
    for v in first_row:
        if isinstance(v, str) and v.strip() != "":
            wellname = v.strip()
            break
    if wellname is None:
        raise RuntimeError(f"文件 {p} 第一行无法识别井名")

    # 
    start_row = None
    for i in range(1, len(raw)):
        row = raw.iloc[i]
        numeric_count = 0
        for v in row:
            try:
                float(str(v))
                numeric_count += 1
            except:
                pass
        if numeric_count >= 3:  # 
            start_row = i
            break

    if start_row is None:
        raise RuntimeError(f"文件 {p} 未找到有效数据行")

    # ---- 读数据部分 ----
    df = pd.read_excel(p, header=None, skiprows=start_row)

    if df.shape[1] < 5:
        raise RuntimeError(f"文件 {p} 数据列数不足 5 列")

    df = df.iloc[:, :5]
    df.columns = ["X", "Y", "MD", "IMP", "TWT"]

    return wellname, df


def map_well_xy_to_inline_xline(
    df: pd.DataFrame,
    points_xy: np.ndarray,
    iline_flat: np.ndarray,
    xline_flat: np.ndarray,
):
    """
    Map well (X,Y) to nearest seismic (iline,xline) using KDTree.
    Returns:
      iline (int array), xline (int array)
    """
    tree = cKDTree(points_xy)
    X = df["X"].astype(float).values
    Y = df["Y"].astype(float).values
    query_pts = np.column_stack([X, Y])
    dist, idx = tree.query(query_pts, k=1)

    iline = iline_flat[idx].astype(int)
    xline = xline_flat[idx].astype(int)

    print(
        "[WELL] XY->iline/xline distance: min=%.3f max=%.3f"
        % (float(dist.min()), float(dist.max()))
    )
    return iline, xline


def well_to_rgt_txt(
    well_path: str,
    meta: dict,
    points_xy: np.ndarray,
    iline_flat: np.ndarray,
    xline_flat: np.ndarray,
    out_dir: str,
):
    """

    """
    from pathlib import Path
    import numpy as np
    import pandas as pd

    # 
    wellname, df = load_well_table_with_name(well_path)  # X, Y, MD, IMP, TWT

    f1 = float(meta["f1"])   # 
    d1 = float(meta["d1"])   # 
    n1 = int(meta["n1"])     # 

    # ---------- 2) XY -> inline, xline ----------
    iline_all, xline_all = map_well_xy_to_inline_xline(
        df, points_xy, iline_flat, xline_flat
    )
    #
    iline_well = int(np.round(np.median(iline_all)))
    xline_well = int(np.round(np.median(xline_all)))

    # ---------- 3) 
    TWT = df["TWT"].astype(float).values
    IMP = df["IMP"].astype(float).values

    # 
    mask_valid = np.isfinite(TWT) & np.isfinite(IMP)
    TWT = TWT[mask_valid]
    IMP = IMP[mask_valid]

    if len(TWT) < 2:
        raise RuntimeError(f"{well_path} 有效井点太少，无法插值")

    # 
    order = np.argsort(TWT)
    TWT = TWT[order]
    IMP = IMP[order]

    # ---------- 4) ----------
    # seismic ：t_grid = f1 + k*d1, k = 0..n1-1
    t_grid = f1 + np.arange(n1) * d1

    # 
    t_min = TWT[0]
    t_max = TWT[-1]
    mask_grid = (t_grid >= t_min) & (t_grid <= t_max)
    t_resamp = t_grid[mask_grid]  # 例如 900, 902, 904...

    # 
    IMP_resamp = np.interp(t_resamp, TWT, IMP)

    # ---------- 5)  DataFrame ----------
    iline_out = np.full_like(t_resamp, iline_well, dtype=int)
    xline_out = np.full_like(t_resamp, xline_well, dtype=int)

    out_df = pd.DataFrame(
        {
            "iline": iline_out,
            "xline": xline_out,
            "TWT": t_resamp,        # 
            "IMP": IMP_resamp,
        }
    )

    # ---------- 6) 
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{wellname}.txt"
    out_df.to_csv(
        out_path,
        sep=" ",
        header=True,   # 
        index=False,
        float_format="%.6f",
    )

    print(f"[WELL] Wrote {out_path}, rows={len(out_df)}")
    print(out_df.head())

def main():
    # 1) Seismic: SEGY -> seis.dat + metadata + KDTree points
    seis_vol, meta, points_xy, iline_flat, xline_flat = load_seismic_with_segysak(
        SEGY_SEIS_PATH
    )
    export_dat(seis_vol, OUT_SEIS_DAT, "seis")
    export_metadata(meta, OUT_METADATA)

    # 2) uf: SEGY -> uf.dat
    uf_vol = load_uf_with_segysak(SEGY_UF_PATH, meta)
    if uf_vol is not None:
        export_dat(uf_vol, OUT_UF_DAT, "uf")

    # 3) Collect all well files
    well_files = list(WELL_FILES)
    if WELL_DIR is not None:
        for p in Path(WELL_DIR).glob("*"):
            if p.suffix.lower() in [".xlsx", ".xls", ".csv", ".txt"]:
                well_files.append(str(p))
    well_files = sorted(set(well_files))

    print("\n[WELL] Files to process:")
    for wf in well_files:
        print("   ", wf)

    # 4) Process each well: -> iline xline isamp IMP
    for wf in well_files:
        p = Path(wf)
        if not p.exists():
            print(f"[WELL] Skip missing file: {wf}")
            continue
        print(f"\n[WELL] Processing {wf}")
        well_to_rgt_txt(
            well_path=wf,
            meta=meta,
            points_xy=points_xy,
            iline_flat=iline_flat,
            xline_flat=xline_flat,
            out_dir=OUT_WELL_DIR,
        )

    print("\n[ALL DONE] Seismic + uf + wells preprocessing finished.")


if __name__ == "__main__":
    main()
