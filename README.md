# PoseNeRF

This repository contains the end-to-end Python implementation of PoseNeRF, a hierarchical NeRF pipeline for camera pose estimation without notebooks.

The current defaults are tuned to be runnable on a laptop with an RTX 2060 by reducing ray batch sizes, model size, render chunk size, and pose-time sampling pressure.

It supports:

- dataset download
- hierarchical NeRF training
- multiview evaluation
- pose estimation on held-out test frames
- refinement from pose-estimated test views
- post-training reruns without retraining

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

The code downloads the official 7-Scenes Chess dataset from Microsoft:

```text
https://download.microsoft.com/download/4/9/4/4945E6F5-16A1-4E28-BD35-82D6EAFB8A9B/chess.zip
```

Download and extract it with:

```bash
python -m posenerf.cli --project-dir ./workspace download-data
```

## Commands

Run the complete pipeline:

```bash
python -m posenerf.cli --project-dir ./workspace full-pipeline --download-data
```

Train only:

```bash
python -m posenerf.cli --project-dir ./workspace train --download-data
```

Evaluate an existing checkpoint:

```bash
python -m posenerf.cli --project-dir ./workspace evaluate --checkpoint ./workspace/outputs/checkpoints/hierarchical_nerf_trained.pth
```

Run everything after training without retraining:

```bash
python -m posenerf.cli --project-dir ./workspace post-train --checkpoint ./workspace/outputs/checkpoints/hierarchical_nerf_trained.pth
```

Force a fresh pose-estimation rerun instead of using cached pose files:

```bash
python -m posenerf.cli --project-dir ./workspace post-train --checkpoint ./workspace/outputs/checkpoints/hierarchical_nerf_trained.pth --no-pose-cache
```

## Outputs

The pipeline stores artifacts under:

```text
workspace/
|- data/
|- outputs/
|  |- checkpoints/
|  |- pose_estimation/
|  |- refinement/
|  \- visualizations/
```

Main checkpoints:

- `outputs/checkpoints/hierarchical_nerf_trained.pth`
- `outputs/checkpoints/hierarchical_nerf_best.pth`
- `outputs/checkpoints/hierarchical_nerf_refined.pth`

Main visualizations:

- `outputs/visualizations/checkpoint_eval_multiview.png`
- `outputs/pose_estimation/hierarchical_nerf_pose_estimation_multiview.png`
- `outputs/refinement/hierarchical_nerf_refinement_multiview.png`

Main metric summaries:

- `outputs/visualizations/evaluation_metrics.json`
- `outputs/pose_estimation/pose_metrics.json`
- `outputs/refinement/refinement_metrics.json`

## Citation Note

If you report results based on the 7-scenes dataset, please cite at least one of the papers mentioned under publications.

- [RelocForest paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RelocForests.pdf)
- [7-Scenes paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/10/glocker2013ismar.pdf)

## Notes

- `post-train` is the command to use once training is already finished.
- `post-train` reuses cached estimated poses by default if they already exist.
- The defaults in this repo are intentionally lighter than the Colab notebook so they fit better on an RTX 2060.
- CUDA is strongly recommended.

