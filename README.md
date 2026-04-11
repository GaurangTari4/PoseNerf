# 🚀 PoseNeRF

Welcome to **PoseNeRF** — a clean, end-to-end implementation of a **hierarchical Neural Radiance Field (NeRF)** pipeline for **camera pose estimation**, built entirely without notebooks.

This project is designed to be:
- 💡 Easy to run  
- 🧩 Modular and extensible  
- 🔁 Fully reproducible  

Whether you're experimenting with NeRFs or building pose estimation systems, this repo provides a complete workflow in one place.

---

## ✨ Features

- 📦 Automatic dataset download  
- 🧠 Hierarchical NeRF training  
- 🎥 Multiview evaluation  
- 📍 Camera pose estimation on unseen frames  
- 🔧 Pose refinement pipeline  
- ⚡ Post-training reruns without retraining  

---

## ⚙️ Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📂 Dataset

This project uses the **7-Scenes Chess dataset**.

Download and extract it automatically:

```bash
python -m posenerf.cli --project-dir ./workspace download-data
```

---

## 🧪 Commands

### 🔄 Run Full Pipeline

Train, evaluate, estimate poses, and refine:

```bash
python -m posenerf.cli --project-dir ./workspace full-pipeline --download-data
```

---

### 🏋️ Train Only

```bash
python -m posenerf.cli --project-dir ./workspace train --download-data
```

---

### 📊 Evaluate a Checkpoint

```bash
python -m posenerf.cli --project-dir ./workspace evaluate \
  --checkpoint ./workspace/outputs/checkpoints/hierarchical_nerf_trained.pth
```

---

### 🔁 Post-Training (No Retraining)

```bash
python -m posenerf.cli --project-dir ./workspace post-train \
  --checkpoint ./workspace/outputs/checkpoints/hierarchical_nerf_trained.pth
```

---

### ♻️ Force Fresh Pose Estimation

```bash
python -m posenerf.cli --project-dir ./workspace post-train \
  --checkpoint ./workspace/outputs/checkpoints/hierarchical_nerf_trained.pth \
  --no-pose-cache
```

---

## 📁 Project Structure

```text
workspace/
├── data/
├── outputs/
│   ├── checkpoints/
│   ├── pose_estimation/
│   ├── refinement/
│   └── visualizations/
```

---

## 🧾 Outputs

### 📌 Checkpoints

- `hierarchical_nerf_trained.pth`
- `hierarchical_nerf_best.pth`
- `hierarchical_nerf_refined.pth`

### 🖼️ Visualizations

- Multiview evaluation results  
- Pose estimation previews  
- Refinement comparisons  

### 📈 Metrics

- `evaluation_metrics.json`  
- `pose_metrics.json`  
- `refinement_metrics.json`  

---

## 📚 Citation Note

If you use results from the 7-Scenes dataset, please cite relevant prior work:

- [RelocForest paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RelocForests.pdf)  
- [7-Scenes paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/10/glocker2013ismar.pdf)  

---

## 📝 Notes

- Use `post-train` after training to iterate quickly  
- Cached pose estimates are reused by default  
- Use `--no-pose-cache` to recompute poses  
- Designed for practical experimentation with manageable resource usage  

---

## 💬 Additional Context

A reference notebook (`.ipynb`) is included for understanding the original workflow and experimentation process.  
However, **this repository is fully script-based and does not depend on notebooks**.

---

## 🧠 Final Thoughts

PoseNeRF aims to make NeRF-based pose estimation **simple, reproducible, and flexible**.

If you're working in 3D vision, camera localization, or neural rendering — this is a great starting point.

Happy experimenting! 🚀
