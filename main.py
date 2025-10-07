import sys
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# Utility functions
def img_to_float_arr(img):
    """Return image as float array in range [0,1]. Supports RGB or L modes."""
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:  # grayscale
        arr = arr[:, :, None]
    arr /= 255.0
    return arr

def float_arr_to_img(arr):
    """Convert float array in [0,1] (H,W,C) to PIL Image."""
    arr = np.clip(arr, 0.0, 1.0)
    arr8 = (arr * 255).astype(np.uint8)
    if arr8.shape[2] == 1:
        return Image.fromarray(arr8[:, :, 0], mode='L')
    return Image.fromarray(arr8, mode='RGB')

def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    maxv = 1.0  # since we normalized to [0,1]
    return 20 * np.log10(maxv / np.sqrt(mse))

# FFT helpers
def fft2_channel(channel):
    return np.fft.fftshift(np.fft.fft2(channel))

def ifft2_channel(freq_domain):
    return np.fft.ifft2(np.fft.ifftshift(freq_domain)).real

def magnitude_spectrum_image(freq_domain):
    mag = np.log1p(np.abs(freq_domain).astype(np.float32))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-12)
    return mag

# Compression strategies
def compress_topk(freq, keep_fraction):
    flat = np.abs(freq).ravel()
    k = int(np.ceil(flat.size * keep_fraction))
    if k <= 0:
        return np.zeros_like(freq)
    thresh = np.partition(flat, -k)[-k]
    mask = np.abs(freq) >= thresh
    return freq * mask

def compress_lowpass(freq, keep_fraction):
    h, w = freq.shape
    area = h * w
    k = int(np.ceil(area * keep_fraction))
    s = int(np.ceil(np.sqrt(k)))
    ch, cw = h // 2, w // 2
    half = s // 2
    mask = np.zeros_like(freq, dtype=bool)
    r0, r1 = max(0, ch - half), min(h, ch + half + (s % 2))
    c0, c1 = max(0, cw - half), min(w, cw + half + (s % 2))
    mask[r0:r1, c0:c1] = True
    return freq * mask

# Main compress / decompress
def process_image(img_arr, keep_fraction=0.05, method='topk', save_prefix='out'):
    H, W, C = img_arr.shape
    reconstructed = np.zeros_like(img_arr)
    stats = {'channels': []}

    for ch in trange(C, desc="Processing channels"):
        channel = img_arr[:, :, ch]
        F = fft2_channel(channel)
        mag_img = magnitude_spectrum_image(F)

        if method == 'topk':
            Fc = compress_topk(F, keep_fraction)
        elif method == 'lowpass':
            Fc = compress_lowpass(F, keep_fraction)
        else:
            raise ValueError("method must be 'topk' or 'lowpass'")

        recon = ifft2_channel(Fc)
        recon = np.clip(recon, 0.0, 1.0)
        reconstructed[:, :, ch] = recon

        kept = np.count_nonzero(np.abs(Fc) > 0)
        total = F.size
        stats['channels'].append({
            'kept': int(kept),
            'total': int(total),
            'fraction_kept': kept / total,
            'psnr': psnr(channel, recon)
        })

        # Save intermediate images
        plt.imsave(f"{save_prefix}_ch{ch}_orig.png", channel, cmap='gray')
        plt.imsave(f"{save_prefix}_ch{ch}_spectrum.png", mag_img, cmap='gray')
        plt.imsave(f"{save_prefix}_ch{ch}_recon.png", recon, cmap='gray')
        plt.imsave(f"{save_prefix}_ch{ch}_spectrum_filtered.png",
                   magnitude_spectrum_image(Fc), cmap='gray')

    stats['global_psnr'] = psnr(img_arr, reconstructed)
    tot_kept = sum(c['kept'] for c in stats['channels'])
    tot_total = sum(c['total'] for c in stats['channels'])
    stats['global_fraction_kept'] = tot_kept / tot_total

    recon_img = float_arr_to_img(reconstructed)
    recon_img.save(f"{save_prefix}_reconstructed.png")
    return reconstructed, stats

# Optional: save sparse representation
def save_sparse_representation(freq_array, filename):
    nz = np.nonzero(np.abs(freq_array) > 0)
    values = freq_array[nz]
    np.savez_compressed(filename,
                        rows=nz[0].astype(np.int32),
                        cols=nz[1].astype(np.int32),
                        real=values.real.astype(np.float32),
                        imag=values.imag.astype(np.float32),
                        shape=np.array(freq_array.shape, dtype=np.int32))

# CLI example
def run_cli(input_path, keep_fractions=(0.01, 0.05, 0.1, 0.25), methods=('topk', 'lowpass')):
    img = Image.open(input_path).convert('RGB')
    arr = img_to_float_arr(img)
    base = os.path.splitext(os.path.basename(input_path))[0]

    for method in methods:
        for k in keep_fractions:
            prefix = f"{base}_{method}_{int(k*100)}pct"
            print(f"Processing {input_path} method={method} keep={k} -> prefix={prefix}")
            recon, stats = process_image(arr, keep_fraction=k, method=method, save_prefix=prefix)
            print(f"  global PSNR: {stats['global_psnr']:.2f} dB, fraction kept: {stats['global_fraction_kept']:.4f}")

            # Save compressed sparse representation for first channel
            F = fft2_channel(arr[:, :, 0])
            Fc = compress_topk(F, k) if method == 'topk' else compress_lowpass(F, k)
            save_sparse_representation(Fc, f"{prefix}_ch0_sparse.npz")
            print(f"  saved images & {prefix}_ch0_sparse.npz\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fft_image_compress.py input.jpg")
        sys.exit(1)
    run_cli(sys.argv[1])
