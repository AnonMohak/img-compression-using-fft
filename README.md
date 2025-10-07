#  FFT Image Compression & Visualization

A simple yet powerful Python tool to **compress and decompress images** using the **Fast Fourier Transform (FFT)** — with visualization of all intermediate stages (spectrum, phase, and reconstruction). Also compares topk and lowpass at different fractions(0.005, 0.01, 0.05, 0.1, 0.25) to see tradeoffs.

---

##  Overview

This project demonstrates how **frequency domain compression** works using FFT.  
Instead of storing pixel values directly, we represent an image using its **frequency components** — and then discard less significant (high-frequency) information to reduce image size while maintaining visual quality.

---

## ⚙️ How It Works

1. **Convert Image → Frequency Domain**
   - The 2D Fast Fourier Transform (FFT) decomposes the image into its frequency components (sine and cosine waves).
   - `np.fft.fft2()` computes this, giving a complex-valued array.
   - Each element represents a frequency — low frequencies are near the center; high frequencies are at the edges.

2. **Visualize Magnitude Spectrum**
   - Using `np.fft.fftshift()`, we center the zero-frequency component.
   - The log of the magnitude gives a clear visualization of which frequencies dominate.

3. **Compression Step**
   - We remove small-magnitude frequency coefficients that contribute little to image quality.
   - Only the top `X%` (e.g., 10%) of coefficients by magnitude are retained.
   - This drastically reduces data while keeping most of the image’s structure intact.

4. **Inverse Transform (Decompression)**
   - The reduced frequency data is converted back into the spatial domain using `np.fft.ifft2()`.
   - The result is a **visually similar but smaller** image, representing the compressed version.

---

##  Why Use FFT Compression?

| Feature | Explanation |
|----------|--------------|
|  **Efficiency** | Most natural images have redundant spatial information — FFT focuses on frequency importance. |
|  **Control** | You can directly adjust compression ratio by choosing how many frequency components to keep. |
|  **Transparency** | See and understand *how compression affects the frequency spectrum* — great for learning! |
|  **Extensible** | Can be extended to color images, DCT (JPEG-style), or hybrid methods. |

---

##  Intermediate Visualizations

When you run the program, you’ll see:
1. Original image  
2. Magnitude spectrum (log scale)  
3. Compressed frequency domain (after zeroing small coefficients)  
4. Reconstructed (decompressed) image  

These help visualize the **tradeoff between compression and quality**.

---

## PSNR (Quality Metric)

PSNR (Peak Signal-to-Noise Ratio) measures reconstruction fidelity:
```md
PSNR = 20 * log10(MAX / sqrt(MSE))
```
 -  MAX = 1.0 (since image normalized to [0,1])

 - MSE = mean squared error

 - Higher PSNR = better reconstruction

| PSNR (dB) | Quality                     |
| --------- | --------------------------- |
| > 40      | Very high (barely any loss) |
| 30–40     | Good quality                |
| 20–30     | Noticeable artifacts        |
| < 20      | Poor                        |

---

##  Saving Compressed Data

 - Each frequency channel can be saved as a sparse representation (.npz) file containing:

 - Indices of non-zero frequencies

 - Real and imaginary values

 - Original shape

 - This allows you to later reconstruct the frequency matrix or analyze compression statistics.

---

## Implementation Details

 - FFT: np.fft.fft2 + np.fft.fftshift

 - IFFT: np.fft.ifft2 + np.fft.ifftshift

 - Top-k: Keeps highest magnitude coefficients

 - Low-pass: Keeps a centered square region of frequencies

 - Visualization: log-scaled magnitude spectrum (log(1 + |F|))

---

## Visual Flow

Image (spatial domain)
        ↓
   2D FFT per channel
        ↓
  Magnitude Spectrum
        ↓
 Apply Filter (Top-k or Low-pass)
        ↓
   Inverse FFT (IFFT)
        ↓
 Reconstructed Image

---

##  Usage

###  Requirements
Make sure you have:
- Python 3.8+
- `numpy`
- `matplotlib`
- `pillow`
- 'tqdm` *(for progress bar)
- `opencv-python` *(optional, for extra image support)*

Install dependencies:
```bash
    pip install numpy matplotlib opencv-python
```

### Run the Script
```bash
   python main.py input.jpg
```
