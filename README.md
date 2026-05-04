# TinyNeRF — PyTorch Implementation

A PyTorch implementation of [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934), based on the simplified TinyNeRF version.

> Trained on the classic Lego scene — reaches ~25–26 dB PSNR after 10,000 iterations on a single GPU.

---

## What is NeRF?

NeRF represents a 3D scene as a neural network that maps a 3D position to color and density. Novel views are synthesized by casting rays through the scene, querying the network at sample points along each ray, and compositing the results using volume rendering.

---

## Architecture

- **Input:** 3D position encoded with positional encoding (L=6) → 39-dim
- **Network:** 8-layer MLP, width 256, skip connection at layer 4
- **Output:** RGB + density (σ)
- **Volume rendering:** alpha compositing with transmittance

### Key Equations

**Positional Encoding:**

$$\gamma(p) = \left( p,\ \sin(2^0 p),\ \cos(2^0 p),\ \ldots,\ \sin(2^{L-1} p),\ \cos(2^{L-1} p) \right)$$

**Alpha and Transmittance:**

$$\alpha_i = 1 - \exp(-\sigma_i \cdot \delta_i) \qquad T_i = \prod_{j < i}(1 - \alpha_j)$$

**Volume Rendering:**

$$C(\mathbf{r}) = \sum_i T_i \cdot \alpha_i \cdot \mathbf{c}_i$$

---

## Results

| Iterations | PSNR |
|---|---|
| 500 | ~21 dB |
| 1,000 | ~23 dB |
| 5,000 | ~25 dB |
| 10,000 | ~25–26 dB |

---

## Setup

```bash
pip install torch numpy matplotlib viser
```

Download the dataset:
```bash
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
```

---

## Usage

Open and run `TinyNeRF.ipynb` in Jupyter. The notebook is structured as:

1. Load dataset (106 images, 100×100, Lego scene)
2. Split into 100 train / 6 test images
3. Define positional encoding, ray generation, volume rendering
4. Train for 10,000 iterations with Adam (lr=5e-4)
5. Evaluate PSNR on test set every 500 iterations
6. Interactive 3D visualization with viser

---

## Interactive Visualization (viser)

The notebook includes an interactive 3D viewer using [viser](https://github.com/nerfstudio-project/viser).

- Displays training camera frustums in the scene
- Renders a point cloud from any viewpoint in real time
- Auto-renders when you stop moving the camera (debounced at 300ms)

If running on a remote machine via SSH/Tailscale:
```python
server = viser.ViserServer(host="0.0.0.0", port=8080)
```
Then open `http://<machine-ip>:8080` in your browser.

---

## Limitations

This is TinyNeRF — a simplified version of the full paper:

- No view-dependent color (only 3D position, not 5D)
- No hierarchical sampling (coarse + fine networks)
- Fixed scene, no unbounded backgrounds

These are the main reasons PSNR plateaus at ~26 dB. Full NeRF reaches 28–31 dB on the same scene.

---

## References

- [NeRF paper](https://arxiv.org/abs/2003.08824) — Mildenhall et al., ECCV 2020
- [Original TinyNeRF colab](https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb)
- [viser](https://github.com/nerfstudio-project/viser) — 3D visualization library
