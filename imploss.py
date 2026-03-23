import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
from typing import Union
from pytorch_msssim import ms_ssim
import torchfilters as gf 

ArrayLike = Union[np.ndarray, Tensor]

def imp2ref(imp: ArrayLike, taxis: int = -1) -> ArrayLike:
    """
    impedance to reflectivity.

    Parameters
    ----------
    imp : ArrayLike
        Impedance data. It could be 1D (a trace), 2D (2d seismic) or 3D (3d seismic).
    taxis : int, optional
        The time axis along which to compute the reflectivity. Default is -1 (last axis).
    """
    ndim = imp.ndim
    if taxis < 0:
        taxis += ndim
    if taxis < 0 or taxis >= ndim:
        raise ValueError(
            f"Invalid time axis {taxis} for data with {ndim} dimensions.")

    if imp.min() <= 0:
        imp = imp - imp.min() + 10

    slicer1 = [slice(None)] * ndim
    slicer2 = [slice(None)] * ndim
    slicer1[taxis] = slice(1, None)
    slicer2[taxis] = slice(0, -1)

    if isinstance(imp, torch.Tensor):
        ref = torch.zeros_like(imp, dtype=imp.dtype, device=imp.device)
    else:
        ref = np.zeros_like(imp, dtype=imp.dtype)
    ref[tuple(slicer1)] = (imp[tuple(slicer1)] - imp[tuple(slicer2)]) / (
        imp[tuple(slicer1)] + imp[tuple(slicer2)] + 1e-8)

    return ref


def ref2seis_torch(ref: Tensor, w: Tensor, taxis: int = -1) -> Tensor:
    """
    reflecivity to seismic. It's recommended to use a short wavelet

    Parameters
    ----------
    ref : torch.Tensor
        Reflectivity data.
    w : torch.Tensor
        wavelet, 1D array.
    """

    if not (isinstance(ref, torch.Tensor) and isinstance(w, torch.Tensor)):
        raise TypeError("ref and w must both be torch.Tensor")

    ndim = ref.ndim
    if taxis < 0:
        taxis += ndim
    if not (0 <= taxis < ndim):
        raise ValueError(
            f"Invalid time axis {taxis} for data with {ndim} dims.")

    if taxis != ndim - 1:
        perm = list(range(ndim))
        perm.pop(taxis)
        perm.append(taxis)
        ref = ref.permute(perm)  # now shape [..., T]
    else:
        perm = None

    orig_shape = ref.shape
    B = int(torch.prod(torch.tensor(orig_shape[:-1], dtype=torch.long)))
    T = orig_shape[-1]

    ref = ref.reshape(B, T).unsqueeze(1)
    w = w.view(-1)
    w = w.flip(0).view(1, 1,
                       -1)  # flip, as F.conv1d is a correlation operation

    K = w.numel()
    pad_left = (K - 1) // 2
    pad_right = K - 1 - pad_left
    ref_pad = F.pad(ref, (pad_left, pad_right))
    if w.dtype != ref.dtype:
        w = w.to(ref.dtype)
    if w.device != ref.device:
        w = w.to(ref.device)

    seis = F.conv1d(ref_pad, w)
    seis = seis.squeeze(1).reshape(orig_shape)

    if perm is None:
        return seis

    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return seis.permute(inv_perm)


def _get_time(duration: float, dt: float, tc: float, sym: bool = True):
    """
    See bruges's wavelets.py. Here, we add tc to 
    represent the center of the time series
    """
    n = int(duration / dt)
    odd = n % 2
    k = int(10**-np.floor(np.log10(dt)))
    # dti = int(k * dt)  # integer dt

    if (odd and sym):
        t = np.arange(n)
    elif (not odd and sym):
        t = np.arange(n + 1)
    elif (odd and not sym):
        t = np.arange(n)
    elif (not odd and not sym):
        t = np.arange(n) - 1

    t -= t[-1] // 2

    # return dti * t / k + tc
    return t * dt + tc


def ricker(f, dt, l=None, duration=None, sym=True, return_t=False):
    """
    Ricker wavelet: Zero-phase wavelet with a central peak and two smaller
    side lobes. Ricker wavelet is symmetric and centered at 0

    .. math::
        A=\left(1-2 \pi^{2} f^{2} t^{2}\right) e^{-\pi^{2} f^{2} t^{2}}
    """
    if l is not None:
        duration = l * dt
    elif duration is None:
        raise ValueError("Either l or duration must be provided")

    if isinstance(f, (list, tuple)):
        f = np.array(f)
    
    if isinstance(f, np.ndarray):
        f = f[:, np.newaxis]
    
    t = _get_time(duration, dt, 0, sym)[np.newaxis]
    w = (1 - 2 * (np.pi * f * t)**2) * np.exp(-(np.pi * f * t)**2)
    w = np.squeeze(w)

    if return_t:
        return w, t
    else:
        return w

def source_indepance_loss(seis: Tensor, recons: Tensor, idx: int=None):
    """
    计算与震源无关的损失函数 (向量化版本)
    
    公式: -(s*r_i)/(||s*r_i||) · (r*s_i)/(||r*s_i||)
    其中:
    - s: 地震数据 (seis)
    - r: 重构数据 (recons) 
    - i: 参考道索引 (idx)
    - *: 卷积操作
    - ||·||: L2范数
    
    这样可以消除子波的影响，因为通过归一化卷积消除了振幅差异
    
    Parameters
    ----------
    seis : torch.Tensor
        地震数据，shape: (batch, channels, height, width) 或 (batch, height, width)
    recons : torch.Tensor  
        重构数据，shape与seis相同
    idx : int, optional
        参考道索引，如果为None则使用中间道

    Returns
    -------
    torch.Tensor
        损失值 (标量)
    """
    batch, channels, width, height = seis.shape
    
    if idx is None:
        idx = width // 2
    if idx >= width or idx < 0:
        raise ValueError(f"idx {idx} is out of range for width {width}")
    
    # 调整维度顺序 (B,C,W,H) -> (W,C,B,H)
    seis = seis.permute(2, 1, 0, 3)
    recons = recons.permute(2, 1, 0, 3)
    
    # 提取参考道 [C, B, H]
    s_ref = seis[idx, :, :, :-1]
    r_ref = recons[idx, :, :, :-1]
    
    # 重塑为卷积输入 [B*C*W, 1, H]
    seis = seis.reshape(width*channels, batch, height)    # [256, 2, 256]
    recons = recons.reshape(width*channels, batch, height)  # [256, 2, 256]
    
    # 卷积核 [B*C, 1, H]
    s_ref = s_ref.reshape(batch * channels, 1, height-1)  # [2,1,256]
    r_ref = r_ref.reshape(batch * channels, 1, height-1)  # [2,1,256]
    
    # 分组卷积：in_shape: (b, c1, h), out_shape: (b, c2, h), kernel: (c2, c1//groups, k), 对通道进行分组
    l1 = F.conv1d(seis, r_ref, groups=batch, padding="same")  # [256,2,256]
    l1 = l1 / (torch.norm(l1, p=2, dim=(0,2), keepdim=True) + 1e-8)
    
    l2 = F.conv1d(recons, s_ref, groups=batch, padding="same")  # [256,2,256]
    l2 = l2 / (torch.norm(l2, p=2, dim=(0,2), keepdim=True) + 1e-8)
    
    # 计算逐样本相似度 [B,]
    dot_product = torch.sum(l1 * l2, dim=(0,2))  # [2,]
    loss = -torch.mean(dot_product)  # 标量

    return loss


def recons_loss(imp, seis, f=35, dt=0.001):
    r = ricker(f, dt, seis.shape[-1])
    r = torch.from_numpy(r).to(imp.device, dtype=imp.dtype)
    recons = ref2seis_torch(imp2ref(imp), r)
    recons = (recons - recons.mean()) / recons.std()
    recons = recons / recons.abs().max()
    return source_indepance_loss(seis, recons) + 1


def recons_loss2(imp: Tensor, seis: Tensor, f=35, dt=0.001):
    r = ricker(f, dt, seis.shape[-1])
    r = torch.from_numpy(r).to(imp.device, dtype=imp.dtype)
    recons = ref2seis_torch(imp2ref(imp), r)
    recons = (recons - recons.mean()) / recons.std()
    recons = recons / recons.abs().max()
    return F.mse_loss(seis, recons)


def recons_loss3(imp, seis, f=35, dt=0.001):
    r = ricker(f, dt, seis.shape[-1])
    r = torch.from_numpy(r).to(imp.device, dtype=imp.dtype)
    recons = ref2seis_torch(imp2ref(imp), r)
    seis = seis * 0.5 + 0.5
    recons = (recons - recons.mean()) / recons.std()
    recons = recons / torch.abs(recons).max() * 0.5 + 0.5
    return 1-ms_ssim(seis, recons, 1.0)



def spectrum_loss(imp, ref, mask=None, dt=0.001):
    B, C, W, H = imp.shape
    imp = gf.highfilter_pad(imp, dt, 10, 1)
    if dt == 0.002:
        padH = 512 
    elif dt == 0.001:
        padH = 768
    else:
        padH = 512
    if H < padH:
        ph = (padH - H) // 2
        ph2 = padH - H - ph
        imp = F.pad(imp, (ph, ph2, 0, 0), mode='constant', value=0)
        assert imp.shape[-1] == padH, f"{imp.shape}"

    spectrum = torch.fft.rfft(imp, dim=-1).abs().mean(dim=(0, 1, 2))[:len(ref)]
    spectrum = 20 * torch.log10(spectrum / spectrum.max())

    if mask is not None:
        spectrum = spectrum[mask]
        ref = ref[mask]

    return F.mse_loss(spectrum, ref.to(spectrum.device))



if __name__ == '__main__':
    a = torch.randn(1, 1, 256, 256).cuda()
    b = torch.randn(1, 1, 256, 256).cuda()

    l = source_indepance_loss(a, a)
    print(l)
