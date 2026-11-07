# Canny Edge Detector from Scratch
**Author:** Joseph Pokuta

**Course:** Intro to Robotics (EE2180) | Michigan Technological University

**Date:** 4/15/25 (Updated 3/22/26)

This project implements multi-stage Canny Edge Detection of an image in Python without relying on `cv2.Canny()`.

<table align="center">
  <tr>
    <th>Input</th>
    <th>Output</th>
  </tr>
  <tr>
    <td><img src="images/IMG_3964.jpg" alt="Input photo" width="350" height="467"></td>
    <td><img src="output/canny_IMG_3964.jpg" alt="Canny edge detection output" width="350" height="467"></td>
  </tr>
</table>

## Implementation
The pipeline processes raw images through four stages to extract edge structure while suppressing noise.

```mermaid
flowchart LR
    A["Input Image"] --> B["Grayscale"]
    B --> C["Gaussian Blur"]
    C --> D["Sobel X / Sobel Y"]
    D --> E["Gradient Magnitude<br/>and Direction"]
    E --> F["Non-Maximum<br/>Suppression"]
    F --> G["Double Threshold"]
    G --> H["Hysteresis"]
    H --> I["Edge Map"]

    C -.->|"blurred_*.png"| O[("output/")]
    D -.->|"sobelX_*.png, sobelY_*.png"| O
    E -.->|"sobelMag_*.png"| O
    F -.->|"thinned_*.png"| O
    H -.->|"final_canny_*.png"| O
```

### Gaussian Blurring
To reduce image noise, I applied a Gaussian filter. The weight $G$ at any pixel $(x, y)$ is calculated using the 2D Gaussian distribution:

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

In my implementation, I utilized `np.meshgrid` to generate the coordinate space and `np.exp` for the exponential decay, allowing for dynamic $\sigma$ tuning.
* **Standard Deviation ($\sigma$):** Controls the "spread." A higher $\sigma$ was used for the **Noisy Cameraman** sample to suppress high-frequency noise.
* **Kernel Size:** 3x3 kernels were used for fine detail, while 5x5 was used for the **High Shadow Lift Bridge** to smooth over shadow gradients.

### Sobel Gradient
Using Sobel-X and Sobel-Y kernels to find the intensity gradients.
* **Magnitude:** $\sqrt{G_x^2 + G_y^2}$
* **Direction:** $\arctan2(G_y, G_x)$

### Non-Maximum Suppression (NMS)
Edges are thinned by comparing a pixel's magnitude to its neighbors along the gradient direction. If the center pixel isn't the local maximum, it is suppressed to zero.

### Double Thresholding & Hysteresis
Pixels are classified into three categories based on `maxRatio` and `minRatio`:
1. **Strong (255):** Above `highThresh`.
2. **Weak (75):** Between `lowThresh` and `highThresh`.
3. **Suppressed (0):** Below `lowThresh`.

**Hysteresis:** Weak pixels are only promoted to Strong if they are connected to an existing Strong pixel within 8 adjacent tiles.

## Project Structure
```
Canny-Edge-Detector/
├── images/                 # Input test images
├── output/                 # Generated images from each pipeline stage
│   ├── blurred_*.png       #   Gaussian blur
│   ├── sobelX_*.png        #   Horizontal gradient (Gx)
│   ├── sobelY_*.png        #   Vertical gradient (Gy)
│   ├── sobelMag_*.png      #   Gradient magnitude
│   ├── thinned_*.png       #   Non-maximum suppression
│   └── final_canny_*.png   #   Final edge map after hysteresis
├── src/
│   ├── canny_detector.py   # Full implementation and pipeline runner
│   └── canny_skeleton.py   # Original skeleton code provided for the project
├── LICENSE
└── README.md
```

## How to Run

### Prerequisites
Ensure the necessary Python libraries: `numpy`, `opencv-python` (for image I/O), and `matplotlib`.
```bash
pip install opencv-python numpy matplotlib
```

### Running the Pipeline
1. Clone the repository.
2. Ensure your test images are in the `/images` folder.
3. Navigate to the source directory and run the script:
```bash
cd src
python canny_detector.py
```

## Results and Tuning
Different image characteristics require specific tuning. I found that the **High-Shadow Lift Bridge** required a larger 5x5 kernel and higher thresholds to avoid false edges caused by shadow boundaries.
| Image Case | Kernel | $\sigma$ | Low Ratio | High Ratio |
| :--- | :---: | :---: | :---: | :---: |
| Sanity Check | 3x3 | 0.5 | 0.05 | 0.15 |
| Noisy Cameraman | 3x3 | 1.5 | 0.05 | 0.50 |
| Lift Bridge | 5x5 | 1.5 | 0.25 | 0.50 |

### Pipeline Progression: Sanity Check
| Original | Gaussian Blur | Sobel Magnitude | Final Canny |
| :---: | :---: | :---: | :---: |
| ![Original](images/sanityCheck.png) | ![Blurred](output/blurred_sanityCheck.png) | ![Sobel Magnitude](output/sobelMag_sanityCheck.png) | ![Final](output/final_canny_sanityCheck.png) |

## Limitations
* **Performance:** Convolution, NMS, and hysteresis use pixel-by-pixel Python loops, so large images are slow. This is why the Lift Bridge and IMG_3964 inputs are resized before processing. Vectorizing with NumPy slicing would speed this up significantly.
* **Single-pass hysteresis:** Weak pixels are only checked against their immediate 8 neighbors in one pass. A weak edge connected to a strong edge only through other weak pixels may be dropped. A flood-fill or iterative approach would trace full edge chains.
* **Quantized gradient direction:** NMS rounds gradient direction to 0°, 45°, 90°, or 135°, which can produce slightly jagged edges on curves. Interpolating between neighbors would give smoother results.
* **Zero-padded borders:** Kernels pad the image with zeros, which can create false gradients along image edges. Reflect or replicate padding would reduce these artifacts.
* **Manual tuning:** Kernel size, $\sigma$, and threshold ratios are tuned by hand per image. There is no automatic threshold selection (e.g. based on median intensity or Otsu's method).
