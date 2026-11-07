# Runtime validation record

## 2026-10-31 environment discovery

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB VRAM.
- NVIDIA driver: 610.88; CUDA UMD version: 13.3.
- No CUDA compiler (`nvcc`) is installed. This is not required for official PyTorch binary wheels.
- Repository-local `.venv` uses Python 3.13.12 and is ignored by Git.
- The initial repository-local environment had no PyTorch or project dependencies.
- A pre-existing user virtual environment also has Python 3.13.12 but no project dependencies.
- The 7-Scenes Chess dataset is not present under the repository. Its official URL is configured in `posenerf.config.CHESS_URL`.

## Installation outcome

PyTorch's official current installation guidance supports Windows Python 3.10--3.14 and offers CUDA 12.6 wheels. The system driver is newer than that CUDA runtime, so no driver or toolkit change is needed.

The official CUDA 12.6 installation completed despite the foreground terminal losing its output. The later CPU fallback log established the actual failure mode: its 126.6 MiB wheel downloaded completely, but the install failed with `WinError 32` because a process held a file in `site-packages\\torch`. This was a transient Windows file lock, not a network, disk, TLS, CUDA, or Python ABI issue.

The active project environment is:

- Python 3.13.12
- Torch 2.14.1+cu126
- TorchVision 0.29.1+cu126
- TorchAudio 2.11.0+cu126
- Torch CUDA runtime 12.6
- CUDA available: true; device: NVIDIA GeForce RTX 3060 Laptop GPU (compute capability 8.6)
- NumPy 2.5.4, Pillow 13.0.0, Matplotlib 3.11.2, TQDM 4.70.1, ImageIO 2.38.1

`pip check` passes.

## Validation status

All six regression tests pass. CPU and CUDA synthetic rendering, backward propagation, checkpoint round-trip, full-image rendering, and RGB-only pose optimization pass with finite outputs. CUDA CPU/GPU maximum output differences were at most `5.96e-7`.

Tiny smoke configuration memory: 16.8 MiB allocated for forward/backward, 18.3 MiB for a 320x240 render, and 19.3 MiB for two pose-optimization iterations. A bounded default-architecture measurement (256 hidden units, 8 layers, 48+48 samples, 128 rays) used 150.1 MiB allocated and 230.0 MiB reserved. These are not full-training memory measurements.

Real-data geometry validation remains blocked by design because the Chess dataset has not been downloaded.
