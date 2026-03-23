# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
import math
from itertools import product
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from typing import Union, List
from julius import bandpass_filter, lowpass_filter, highpass_filter


def _gaussian_core(img: Tensor, kernel: Tensor) -> Tensor:
    """
    使用一维高斯核在每个维度上分别对输入张量进行高斯滤波。

    参数：
    - img (Tensor): 输入张量，形状为 (N, C, D1, D2, ..., Dn)。
    - kernel (Tensor): 一维高斯核，形状为 (kernel_size,)。

    返回：
    - Tensor: 滤波后的张量，形状与输入相同。
    """
    # NOTE: 计算前请对img做padding, 这个函数不会添加padding
    dims = img.ndim - 2  # 空间维度的数量

    for dim in range(dims):
        # 将要处理的维度移到最后一维
        permute_order = list(range(img.dim()))
        permute_order.append(permute_order.pop(2))
        img = img.permute(*permute_order)

        img_shape = list(img.shape)
        img = img.reshape(-1, 1, img.shape[-1])  # 形状为 (N * C * 其他维度, L)
        # 准备卷积核
        kernel1d = kernel[dim]  # 形状为 (1, 1, kernel_size)
        # 应用一维卷积
        img = F.conv1d(img, kernel1d)
        img_shape[-1] = img.shape[-1]
        # 恢复原始形状
        img = img.view(*img_shape)

    return img


def _get_gaussian_kernel_1d(kernel_size, sigma, dtype, device):
    """
    生成每个维度的一维高斯核列表。

    参数：
    - kernel_size (list or tuple): 每个维度的核大小。
    - sigma (list or tuple): 每个维度的标准差。
    - dtype: 数据类型。
    - device: 设备。

    返回：
    - kernels (list of Tensor): 每个维度对应的高斯核，形状为 (1, 1, L_i)。
    """
    kernels = []
    for size, s in zip(kernel_size, sigma):
        # 创建坐标
        x = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2

        # 计算高斯核
        kernel = torch.exp(-0.5 * (x / s)**2)
        kernel = kernel / kernel.sum()

        # 调整形状为 (1, 1, L_i)
        kernel = kernel.view(1, 1, -1)

        kernels.append(kernel)
    return kernels


def _preprocess(img, kernel_size, sigma):
    # Ensure kernel_size and sigma are lists
    ndim = min(img.ndim, 3)
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * ndim
    if isinstance(sigma, (int, float)):
        sigma = [sigma] * ndim
    ndim = len(kernel_size)
    shape = img.shape
    img = img.reshape(-1, 1, *shape[-ndim:])

    dtype_torch = img.dtype
    device = img.device

    # Generate the Gaussian kernel
    kernel = _get_gaussian_kernel_1d(
        kernel_size,
        sigma,
        dtype=dtype_torch,
        device=device,
    )

    return img, kernel


# 只对3D数据做分块
def _gaussian_filters3d_tile(
    img: Tensor,
    kernel: List[Tensor],
    tile_size: int,
    padding: List,
):
    assert tile_size >= max(
        padding
    ), f"tile_size ({tile_size}) must larger than the max padding size ({max(padding)})"

    _, _, h, w, t = img.shape
    nh = math.ceil(h / tile_size)
    nw = math.ceil(w / tile_size)
    nt = math.ceil(t / tile_size)

    pkl, pkr, pjl, pjr, pil, pir = padding

    out = torch.zeros_like(img).to(img.device, dtype=img.dtype)

    for i, j, k in product(np.arange(nh), np.arange(nw), np.arange(nt)):
        ib, ie = i * tile_size, min((i + 1) * tile_size, h)
        jb, je = j * tile_size, min((j + 1) * tile_size, w)
        kb, ke = k * tile_size, min((k + 1) * tile_size, t)

        ibp, iep = ib - pil, ie + pir
        jbp, jep = jb - pjl, je + pjr
        kbp, kep = kb - pkl, ke + pkr

        pad = [0, 0, 0, 0, 0, 0]
        if ibp < 0:
            pad[4] = pil
            ibp = 0
        if jbp < 0:
            pad[2] = pjl
            jbp = 0
        if kbp < 0:
            pad[0] = pkl
            kbp = 0
        if iep > h:
            pad[5] = iep - h
            iep = h
        if jep > w:
            pad[3] = jep - w
            jep = w
        if kep > t:
            pad[1] = kep - t
            kep = t

        subd = img[:, :, ibp:iep, jbp:jep, kbp:kep]

        if max(pad) > 0:
            subd = F.pad(subd, pad, mode='reflect')

        out[:, :, ib:ie, jb:je, kb:ke] = _gaussian_core(subd, kernel)

    return out


def gaussian_filter(
    img: Union[Tensor, np.ndarray],
    kernel_size: Union[int, List[int]],
    sigma: Union[float, List[float]],
    tile_size: int = -1,
) -> Union[Tensor, np.ndarray]:
    """
    Apply a Gaussian filter to the input tensor using specified kernel_size and sigma.

    Parameters
    ------------
    - img (Tensor or numpy.ndarray): Input tensor.
    - kernel_size (int or list): Kernel size for each dimension.
    - sigma (float or list): Standard deviation for each dimension.
    - dtype (str): Data type for computation ('fp32' or 'fp16').
    - device (str): Device to perform computation ('cpu' or 'cuda').

    Returns
    ---------
    - Tensor or numpy.ndarray: Blurred tensor with the same shape as the input.
    """
    ndim = img.ndim
    shape = img.shape

    npout = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        npout = True

    img, kernel = _preprocess(img, kernel_size, sigma)

    # Compute padding
    kernel_size = [k.shape[-1] for k in kernel]
    padding = []
    for size in reversed(kernel_size):
        pad_left = size // 2
        pad_right = size - 1 - pad_left
        padding.extend([pad_left, pad_right])

    # Call the core function
    if tile_size > 0 and ndim == 3:
        img = _gaussian_filters3d_tile(img, kernel, tile_size, padding)
    else:
        img = F.pad(img, padding, mode='reflect')
        img = _gaussian_core(img, kernel)

    result = result.reshape(shape)
    if npout:
        result = result.detach().cpu().numpy()

    return result


def gaussian_filter_scipy(
    img: Union[Tensor, np.ndarray],
    sigma: Union[float, List[float]],
    tile_size: int = -1,
    truncate: float = 4.0,
) -> Union[Tensor, np.ndarray]:
    """
    Mimic SciPy's gaussian_filter by computing the kernel size from sigma and truncate.

    Parameters:
    - img (Tensor or numpy.ndarray): Input tensor or array.
    - sigma (float or list): Standard deviation(s) for the Gaussian kernel.
    - truncate (float): Truncate the filter at this many standard deviations (default 4.0).
    - dtype (str): Data type for computation ('fp32' or 'fp16').
    - device (str): Device to perform computation ('cpu' or 'cuda').

    Returns:
    - Tensor or numpy.ndarray: Blurred tensor or array with the same shape as the input.
    """
    ndim = img.ndim
    shape = img.shape
    npout = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        npout = True

    if isinstance(sigma, (int, float)):
        sigma = [sigma] * img.ndim

    # Compute the kernel size for each dimension
    kernel_size = []
    for s in sigma:
        # Compute the radius
        radius = int(truncate * s + 0.5)
        size = 2 * radius + 1
        kernel_size.append(size)

    img, kernel = _preprocess(img, kernel_size, sigma)

    # Compute padding
    kernel_size = [k.shape[-1] for k in kernel]
    padding = []
    for size in reversed(kernel_size):
        pad_left = size // 2
        pad_right = size - 1 - pad_left
        padding.extend([pad_left, pad_right])

    if tile_size > 0 and ndim == 3:
        img = _gaussian_filters3d_tile(img, kernel, tile_size, padding)
    else:
        img = F.pad(img, padding, mode='reflect')
        img = _gaussian_core(img, kernel)

    img = img.reshape(shape)
    if npout:
        img = img.detach().cpu().numpy()

    return img


def _filter_preprocess(d, device='cuda'):
    npout = False
    if isinstance(d, np.ndarray):
        npout = True
        d = torch.from_numpy(d)
    d = d.to(device)
    return d.reshape(-1, d.shape[-1]), npout


def _bandfilter_pad(d, dt, lowcut, high_cut, zeros, device='cuda'):
    """
    bandpass filter with padding to avoid boundary effects
    """
    shape = d.shape
    d, npout = _filter_preprocess(d, device)
    w, h = d.shape

    pad_size = min(h // 2, 50)
    d = F.pad(d, (pad_size, pad_size, 0, 0), mode='constant', value=0)

    d = bandpass_filter(d, dt * lowcut, dt * high_cut, zeros=zeros)
    d = d[:, pad_size:pad_size + h].reshape(shape)

    if npout:
        d = d.detach().cpu().numpy()
    return d


def bandfilter_pad(d, dt, lowcut, high_cut, zeros, device=None, split=False):
    if device is None:
        device = d.device if isinstance(d, torch.Tensor) else 'cpu'
    if split and d.ndim > 2:
        shape = d.shape
        d = d.reshape(-1, shape[-2], shape[-1])
        for i in range(d.shape[0]):
            d[i] = _bandfilter_pad(d[i], dt, lowcut, high_cut, zeros, device)
        d = d.reshape(shape)
    else:
        d = _bandfilter_pad(d, dt, lowcut, high_cut, zeros, device)
    return d


def _lowfilter_pad(d, dt, lowcut, zeros, device='cuda'):
    """
    lowpass filter with padding to avoid boundary effects
    """
    shape = d.shape
    d, npout = _filter_preprocess(d, device)
    w, h = d.shape

    pad_size = min(h // 2, 50)
    d = F.pad(d, (pad_size, pad_size, 0, 0), mode='constant', value=0)
    d = lowpass_filter(d, dt * lowcut, zeros=zeros)
    d = d[:, pad_size:pad_size + h].reshape(shape)

    if npout:
        d = d.detach().cpu().numpy()
    return d


def lowfilter_pad(d, dt, lowcut, zeros, device=None, split=False):
    if device is None:
        device = d.device if isinstance(d, torch.Tensor) else 'cpu'
    if split and d.ndim > 2:
        shape = d.shape
        d = d.reshape(-1, shape[-2], shape[-1])
        for i in range(d.shape[0]):
            d[i] = _lowfilter_pad(d[i], dt, lowcut, zeros, device)
        d = d.reshape(shape)
    else:
        d = _lowfilter_pad(d, dt, lowcut, zeros, device)
    return d


def _highfilter_pad(d, dt, high_cut, zeros, device='cuda'):
    """
    highpass filter with padding to avoid boundary effects
    """
    shape = d.shape
    d, npout = _filter_preprocess(d, device)
    w, h = d.shape

    pad_size = min(h // 2, 50)
    d = F.pad(d, (pad_size, pad_size, 0, 0), mode='constant', value=0)

    d = highpass_filter(d, dt * high_cut, zeros=zeros)
    d = d[:, pad_size:pad_size + h].reshape(shape)

    if npout:
        d = d.detach().cpu().numpy()
    return d


def highfilter_pad(d, dt, high_cut, zeros, device=None, split=False):
    if device is None:
        device = d.device if isinstance(d, torch.Tensor) else 'cpu'
    if split and d.ndim > 2:
        shape = d.shape
        d = d.reshape(-1, shape[-2], shape[-1])
        for i in range(d.shape[0]):
            d[i] = _highfilter_pad(d[i], dt, high_cut, zeros, device)
        d = d.reshape(shape)
    else:
        d = _highfilter_pad(d, dt, high_cut, zeros, device)
    return d


def _filter_low(d, dt, k=-120, lowcut=75, zeros=5, device='cuda'):
    """
    对数据 d 的从索引 k 开始的部分进行低通滤波，并处理边缘拼接。

    参数：
    - d: 输入数据数组，形状为 (n_samples, n_features)
    - dt: 采样间隔
    - k: 滤波开始的索引位置（可以为负值）
    - lowcut: 低通滤波的截止频率
    - zeros: 滤波器的零点数量
    返回：
    - d: 经过滤波和拼接处理后的数据数组
    """
    shape = d.shape
    d, npout = _filter_preprocess(d, device)
    w, h = d.shape

    # 调整负索引
    if k < 0:
        k += h

    # 确保索引不越界
    k = max(0, min(k, h))

    # 定义过渡区域宽度
    blend_w = 20
    half_blend = blend_w // 2

    # 计算过渡区域的起始和结束索引
    start_blend = max(k - half_blend, 0)
    end_blend = min(k + half_blend, h)

    # 提取需要滤波的部分
    start_filt = start_blend
    d_filt = d[:, start_filt - 20:]

    # 对提取的部分进行低通滤波
    d_filt = lowfilter_pad(d_filt, dt, lowcut, zeros, device=device)[:, 20:]

    blend_range = end_blend - start_blend
    if blend_range > 0:
        weights = torch.linspace(0, 1, blend_range).unsqueeze(0).to(d.device, dtype=d.dtype) # yapf: disable

        d[:, start_blend:end_blend] = (1 - weights) * d[:, start_blend:end_blend] + \
                                        weights * d_filt[:, :blend_range]

    d[:, end_blend:] = d_filt[:, blend_range:]

    d = d.reshape(shape)
    if npout:
        d = d.detach().cpu().numpy()
    return d


def filter_low(d, dt, k=-120, lowcut=75, zeros=5, device=None, split=False):
    if device is None:
        device = d.device if isinstance(d, torch.Tensor) else 'cpu'
    if split and d.ndim > 2:
        shape = d.shape
        d = d.reshape(-1, shape[-2], shape[-1])
        for i in range(d.shape[0]):
            d[i] = _filter_low(d[i], dt, k, lowcut, zeros, device)
        d = d.reshape(shape)
    else:
        d = _filter_low(d, dt, k, lowcut, zeros, device)
    return d


def fftNd(d, dt=0.002, log=True, fmax=150):
    """
    d: input data, shape = (..., nt)
    dt: time interval
    log: whether to return the result in logarithmic scale
    fmax: maximum frequency to consider
    This function uses numpy or torch based on the type of the input data
    to calculate the FFT spectrum and return the frequency array and the 
    average FFT spectrum.
    """
    # Determine whether to use numpy or torch
    if isinstance(d, torch.Tensor):
        use_torch = True
        backend = torch
    else:
        use_torch = False
        backend = np

    d = d.reshape(-1, d.shape[-1])
    nt = d.shape[-1]

    # fmt: off
    if nt < 512:
        pad_size = (512 - nt) // 2
        if use_torch:
            d = torch.nn.functional.pad(d, (pad_size, pad_size), mode='constant', value=0)
        else:
            d = np.pad(d, ((0, 0), (pad_size, pad_size)), mode='constant', constant_values=0)

    if use_torch:
        fftd = torch.fft.rfft(d, dim=-1)
        freq = torch.fft.rfftfreq(d.shape[-1], dt)
    else:
        fftd = np.fft.rfft(d, axis=-1)
        freq = np.fft.rfftfreq(d.shape[-1], dt)

    avg_fftd = backend.abs(fftd).mean(axis=0)

    if log:
        avg_fftd = 20 * backend.log10(avg_fftd / avg_fftd.max())
    else:
        avg_fftd = avg_fftd / avg_fftd.max()

    mask = freq <= fmax
    freq = freq[mask]
    avg_fftd = avg_fftd[mask]

    if use_torch:
        freq = freq.detach().cpu().numpy()
        avg_fftd = avg_fftd.detach().cpu().numpy()

    return freq, avg_fftd
