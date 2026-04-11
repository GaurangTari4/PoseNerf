# 🚀 PoseNeRF

## 🚀 Introduction

**PoseNeRF** is an end-to-end implementation of a **hierarchical Neural
Radiance Field (NeRF)** framework designed for **joint scene
representation and camera pose estimation**.

Unlike standard NeRF pipelines that assume known camera poses, PoseNeRF
explores a more realistic and challenging setting: **recovering accurate
camera poses directly from images while simultaneously leveraging a
learned radiance field**.

The core idea is to: - First learn a **coarse-to-fine hierarchical NeRF
representation** of a scene\
- Then use the learned representation as a **geometric prior** to infer
poses for unseen views\
- Finally refine both poses and renderings through iterative
optimization

This repository translates an experimental notebook-based workflow into
a **fully modular, script-driven pipeline**, making it easier to: -
Reproduce results consistently\
- Run large-scale experiments\
- Integrate into research or production workflows

The pipeline reflects a practical research setup: - Controlled training
with reduced computational overhead\
- Explicit separation of training, evaluation, pose estimation, and
refinement stages\
- Support for re-running downstream tasks without retraining

------------------------------------------------------------------------

## ✨ Features

-   📦 Automatic dataset download\
-   🧠 Hierarchical NeRF training\
-   🎥 Multiview evaluation\
-   📍 Camera pose estimation on unseen frames\
-   🔧 Pose refinement pipeline\
-   ⚡ Post-training reruns without retraining

------------------------------------------------------------------------

## ⚙️ Setup

``` bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📂 Dataset

This project uses the **7-Scenes Chess dataset**.

``` bash
python -m posenerf.cli --project-dir ./workspace download-data
```

------------------------------------------------------------------------

## 🧪 Commands

### 🔄 Full Pipeline

``` bash
python -m posenerf.cli --project-dir ./workspace full-pipeline --download-data
```

### 🏋️ Train

``` bash
python -m posenerf.cli --project-dir ./workspace train --download-data
```

### 📊 Evaluate

``` bash
python -m posenerf.cli --project-dir ./workspace evaluate \
  --checkpoint ./workspace/outputs/checkpoints/hierarchical_nerf_trained.pth
```

### 🔁 Post-train

``` bash
python -m posenerf.cli --project-dir ./workspace post-train \
  --checkpoint ./workspace/outputs/checkpoints/hierarchical_nerf_trained.pth
```

------------------------------------------------------------------------

## 📁 Project Structure

    workspace/
    ├── data/
    ├── outputs/
    │   ├── checkpoints/
    │   ├── pose_estimation/
    │   ├── refinement/
    │   └── visualizations/

------------------------------------------------------------------------

## 📊 Sample Results

-   **Best refinement step:** 200\
-   **Average PSNR:** 17.69 dB\
-   **Depth MAE:** 0.336 m

### Observations

-   Refinement improves sharpness\
-   Stable PSNR gains\
-   Coarse geometry recovered well

------------------------------------------------------------------------

## 📚 Citation

-   RelocForest\
-   7-Scenes dataset

------------------------------------------------------------------------

## 🧠 Final Thoughts

PoseNeRF provides a clean, reproducible baseline for **NeRF-based pose
estimation**.

Happy experimenting 🚀
