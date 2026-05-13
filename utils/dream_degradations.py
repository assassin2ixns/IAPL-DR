import random

import torch
import torch.nn.functional as F


def get_dataset_mean_std(dataset):
    if dataset in ['GenImage', 'Chameleon_SD']:
        mean = [0.481, 0.458, 0.408]
        std = [0.269, 0.261, 0.276]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return mean, std


def _mean_std_tensors(dataset, x):
    mean, std = get_dataset_mean_std(dataset)
    mean = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return mean, std


def denormalize(x, dataset):
    mean, std = _mean_std_tensors(dataset, x)
    return x * std + mean


def normalize(x, dataset):
    mean, std = _mean_std_tensors(dataset, x)
    return (x - mean) / std


def clamp01(x):
    return x.clamp(0.0, 1.0)


def gaussian_blur_tensor(img, kernel_size=3):
    assert kernel_size in [3, 5]
    dtype = img.dtype
    img_f = img.float()
    coords = torch.arange(kernel_size, dtype=img_f.dtype, device=img.device) - (kernel_size - 1) / 2.0
    sigma = 0.8 if kernel_size == 3 else 1.2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(img.shape[1], 1, -1, -1)
    pad = kernel_size // 2
    out = F.conv2d(F.pad(img_f, (pad, pad, pad, pad), mode='reflect'), kernel, groups=img.shape[1])
    return out.to(dtype=dtype)


def down_up_tensor(img, scale):
    dtype = img.dtype
    h, w = img.shape[-2:]
    down_h = max(1, int(round(h * scale)))
    down_w = max(1, int(round(w * scale)))
    img_f = img.float()
    out = F.interpolate(img_f, size=(down_h, down_w), mode='bilinear', align_corners=False)
    out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
    return out.to(dtype=dtype)


def quantize_tensor(img, levels):
    levels = max(1, int(levels))
    return torch.round(img * levels) / levels


def jpeg_like_tensor(img, quality=None):
    q = int(quality) if quality is not None else random.choice([50, 75, 90])
    levels = int(32 + q / 100.0 * 223)
    out = quantize_tensor(img, levels)
    if q <= 75:
        scale = random.choice([0.75, 0.875]) if quality is None else (0.75 if q <= 60 else 0.875)
        out = down_up_tensor(out, scale=scale)
    if q <= 60:
        out = gaussian_blur_tensor(out, kernel_size=3)
    return clamp01(out)


def webp_like_tensor(img):
    out = quantize_tensor(img, levels=96)
    out = gaussian_blur_tensor(out, kernel_size=3)
    out = down_up_tensor(out, scale=0.875)
    return clamp01(out)


def apply_named_degradation(x, name, dataset, quality=None):
    if name in ['none', 'identity']:
        return x

    dtype = x.dtype
    img = clamp01(denormalize(x, dataset)).to(dtype=dtype)

    if name == 'jpeg':
        out = jpeg_like_tensor(img, quality=quality)
    elif name == 'jpeg50':
        out = jpeg_like_tensor(img, quality=50)
    elif name == 'jpeg75':
        out = jpeg_like_tensor(img, quality=75)
    elif name == 'jpeg90':
        out = jpeg_like_tensor(img, quality=90)
    elif name == 'resize':
        out = down_up_tensor(img, scale=0.5)
    elif name == 'blur':
        out = gaussian_blur_tensor(img, kernel_size=5)
    elif name == 'quant':
        out = quantize_tensor(img, levels=32)
    elif name == 'webp':
        out = webp_like_tensor(img)
    else:
        raise ValueError('Unsupported DREAM degradation: {}'.format(name))

    out = clamp01(out).to(dtype=dtype)
    return normalize(out, dataset).to(dtype=dtype)


def make_train_degradation_views(x, args):
    num_views = max(0, int(getattr(args, 'dream_num_train_views', 0)))
    pool = [name for name in getattr(args, 'dream_degradation_pool', []) if name not in ['none', 'identity']]
    if len(pool) == 0:
        pool = ['jpeg']

    views = []
    with torch.no_grad():
        for _ in range(num_views):
            name = random.choice(pool)
            views.append(apply_named_degradation(x, name, args.dataset))
    return views


def make_eval_degradation(x, args):
    name = getattr(args, 'dream_eval_degradation', 'none')
    if name == 'none':
        return x
    with torch.no_grad():
        return apply_named_degradation(x, name, args.dataset)
