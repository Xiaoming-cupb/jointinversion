
import os
import yaml
import numpy as np
import torch
from tqdm import tqdm
from net_torch import FullyConvModel

# -----------------------------
# Utils: well stats for denorm
# -----------------------------
def well_stats_from_txt_folder(well_dir: str, imp_col: int = -1, skip_header: bool = True,
                               drop_zero: bool = True, eps: float = 1e-8):
    """
    Read all *.txt in well_dir. Each txt is like:
      iline xline TWT IMP
      ...
    Returns:
      mean_imp, std_imp, scale_s
    where scale_s = max(|(imp-mean)/std|) (i.e., absmax after z-score).
    """
    well_dir = os.path.abspath(well_dir)
    if not os.path.isdir(well_dir):
        raise FileNotFoundError(f"well_dir not found: {well_dir}")

    vals = []
    n_files = 0

    for fn in sorted(os.listdir(well_dir)):
        if not fn.lower().endswith(".txt"):
            continue
        fp = os.path.join(well_dir, fn)
        n_files += 1

        # robust load: try skipping header first
        arr = None
        if skip_header:
            try:
                arr = np.loadtxt(fp, skiprows=1)
            except Exception:
                arr = None
        if arr is None:
            arr = np.loadtxt(fp)

        if arr.ndim == 1:
            arr = arr[None, :]

        if arr.shape[1] < (abs(imp_col) if imp_col < 0 else imp_col + 1):
            raise ValueError(f"{fp} has shape {arr.shape}, cannot take imp_col={imp_col}")

        imp = arr[:, imp_col].astype(np.float32)
        imp = imp[np.isfinite(imp)]
        if drop_zero:
            imp = imp[imp != 0]
        if imp.size > 0:
            vals.append(imp)

    if n_files == 0:
        raise RuntimeError(f"No .txt well files found in: {well_dir}")
    if len(vals) == 0:
        raise RuntimeError(f"No valid IMP samples found in: {well_dir}")

    imp_all = np.concatenate(vals)
    mean = float(imp_all.mean())
    std = float(imp_all.std() + eps)

    z = (imp_all - mean) / std
    scale_s = float(np.max(np.abs(z)) + eps)

    print(f"[WELL STATS] files={n_files}, N={imp_all.size}")
    print(f"[WELL STATS] mean={mean:.6f}, std={std:.6f}, z_absmax(scale_s)={scale_s:.6f}")
    return mean, std, scale_s


def main():
    # ====================== Load YAML ======================
    with open('predict.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # ====================== Params ======================
    CUDA_DEVICE = str(config['device']['cuda_device'])
    NT = int(config['data']['nt'])
    NX = int(config['data']['nx'])
    NI = int(config['data']['ni'])
    SEIS_PATH = config['data']['seis_path']
    INIT_PATH = config['data']['init_path']
    NORMALIZE_DATA = bool(config['data'].get('normalize', True))
    MODEL_PATH = config['model']['model_path']
    OUTPUT_PATH = config['output']['output_path']

    # ---- Denorm from wells (optional) ----
    DENORM_IMP = bool(config['data'].get('denorm_imp', True))
    WELL_DIR = config['data'].get('well_txt_dir', None)
    WELL_IMP_COL = int(config['data'].get('well_imp_col', -1))  # default: last column
    WELL_SKIP_HEADER = bool(config['data'].get('well_skip_header', True))
    WELL_DROP_ZERO = bool(config['data'].get('well_drop_zero', True))

    # ====================== Device ======================
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] CUDA_VISIBLE_DEVICES={CUDA_DEVICE}, device={device}")

    # ====================== Load data ======================
    seis = np.fromfile(SEIS_PATH, np.float32).reshape(NI, NX, NT)
    init = np.fromfile(INIT_PATH, np.float32).reshape(NI, NX, NT)

    seist = torch.from_numpy(seis).unsqueeze(1).to(device)
    initt = torch.from_numpy(init).unsqueeze(1).to(device)

    # ====================== Normalize inputs ======================
    if NORMALIZE_DATA:
        seist = (seist - seist.mean()) / (seist.std() + 1e-8)
        seist = seist / (seist.abs().max() + 1e-8)

        initt = (initt - initt.mean()) / (initt.std() + 1e-8)
        initt = initt / (initt.abs().max() + 1e-8)

    # ====================== Load model ======================
    model = FullyConvModel()
    params = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(params)
    model.eval()
    model = model.to(device)

    # ====================== Predict ======================
    out = torch.zeros((NI, NX, NT), dtype=torch.float32)
    with torch.no_grad():
        for i in tqdm(range(NI), desc="Predict", unit="iline"):
            inp = torch.concat([seist[i:i+1], initt[i:i+1]], dim=1)  # [1,2,NX,NT]
            out[i] = model(inp).detach().cpu()

    arr = out.numpy().astype(np.float32)

    # ====================== Denormalize IMP (optional) ======================
    if DENORM_IMP:
        if WELL_DIR is None:
            raise ValueError("denorm_imp=True but config['data']['well_txt_dir'] is not set.")
        mean_imp, std_imp, scale_s = well_stats_from_txt_folder(
            WELL_DIR,
            imp_col=WELL_IMP_COL,
            skip_header=WELL_SKIP_HEADER,
            drop_zero=WELL_DROP_ZERO
        )
        # inverse of: x_norm = ((x-mean)/std) / scale_s
        # arr = (arr * scale_s) * std_imp + mean_imp
        # absmax_pred = np.max(np.abs(arr)) + 1e-8
        # arr = (arr /absmax_pred) * std_imp + mean_imp
        arr = arr * std_imp + mean_imp



        # # save stats for reproducibility
        # stats_path = OUTPUT_PATH + ".denorm_stats.npz"
        # os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        # np.savez(stats_path, mean=mean_imp, std=std_imp, scale_s=scale_s)
        # print(f"[DENORM] saved stats: {stats_path}")
    else:
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

    # ====================== Save ======================
    arr.tofile(OUTPUT_PATH)
    print(f"Done! Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
