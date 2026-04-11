from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch


CHESS_URL = (
    "https://download.microsoft.com/download/4/9/4/"
    "4945E6F5-16A1-4E28-BD35-82D6EAFB8A9B/chess.zip"
)


@dataclass
class SceneConfig:
    orig_w: int = 640
    orig_h: int = 480
    orig_fx: float = 585.0
    orig_fy: float = 585.0
    orig_cx: float = 320.0
    orig_cy: float = 240.0
    downsample: int = 2
    near: float = 0.1
    far: float = 4.5
    use_opencv_camera_convention: bool = True

    @property
    def width(self) -> int:
        return self.orig_w // self.downsample

    @property
    def height(self) -> int:
        return self.orig_h // self.downsample

    @property
    def fx(self) -> float:
        return self.orig_fx / self.downsample

    @property
    def fy(self) -> float:
        return self.orig_fy / self.downsample

    @property
    def cx(self) -> float:
        return self.orig_cx / self.downsample

    @property
    def cy(self) -> float:
        return self.orig_cy / self.downsample


@dataclass
class TrainConfig:
    max_train_images: int = 96
    max_test_images: int = 12
    batch_rays: int = 1024
    num_steps: int = 12000
    lr: float = 2.0e-4
    num_coarse_samples: int = 48
    num_fine_samples: int = 48
    rgb_loss_weight: float = 1.0
    depth_loss_weight: float = 0.10
    val_render_interval: int = 2000
    checkpoint_every: int = 500
    val_frame_count: int = 4
    use_amp: bool = True
    resume: bool = True

    @property
    def num_samples(self) -> int:
        return self.num_coarse_samples + self.num_fine_samples


@dataclass
class ModelConfig:
    hidden: int = 256
    layers: int = 8
    pos_freqs: int = 10
    dir_freqs: int = 4
    render_chunk: int = 512


@dataclass
class PoseConfig:
    iters: int = 1000
    batch_rays: int = 512
    lr: float = 3.0e-3
    rgb_loss_weight: float = 1.0
    depth_loss_weight: float = 0.20
    init_rot_perturb_deg: float = 3.0
    init_trans_perturb: tuple[float, float, float] = (0.03, -0.02, 0.03)
    frame_count: int = 4
    render_coarse_samples: int = 32
    render_fine_samples: int = 32

    def init_trans_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.init_trans_perturb, dtype=torch.float32, device=device)


@dataclass
class RefinementConfig:
    steps: int = 300
    batch_rays: int = 1024
    lr: float = 1.0e-5
    depth_loss_weight: float = 0.30
    train_images: int = 48
    checkpoint_every: int = 100
    eval_every: int = 100
    frame_count: int = 4
    resume: bool = True


@dataclass
class PathsConfig:
    project_dir: Path
    data_dir: Path
    outputs_dir: Path
    checkpoints_dir: Path
    visualizations_dir: Path
    pose_dir: Path
    refinement_dir: Path
    scene_dir: Path


@dataclass
class ExperimentConfig:
    seed: int = 42
    scene: SceneConfig = field(default_factory=SceneConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    refine: RefinementConfig = field(default_factory=RefinementConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def signature(self) -> Dict[str, Any]:
        return {
            "profile": "laptop-rtx2060",
            "downsample": self.scene.downsample,
            "max_train_images": self.train.max_train_images,
            "batch_rays": self.train.batch_rays,
            "num_steps": self.train.num_steps,
            "lr": self.train.lr,
            "near": self.scene.near,
            "far": self.scene.far,
            "num_coarse_samples": self.train.num_coarse_samples,
            "num_fine_samples": self.train.num_fine_samples,
            "rgb_loss_weight": self.train.rgb_loss_weight,
            "depth_loss_weight": self.train.depth_loss_weight,
            "model_hidden": self.model.hidden,
            "model_layers": self.model.layers,
            "pos_freqs": self.model.pos_freqs,
            "dir_freqs": self.model.dir_freqs,
            "render_chunk": self.model.render_chunk,
            "val_frame_count": self.train.val_frame_count,
        }


def build_paths(root: Path) -> PathsConfig:
    project_dir = root
    data_dir = project_dir / "data"
    outputs_dir = project_dir / "outputs"
    checkpoints_dir = outputs_dir / "checkpoints"
    visualizations_dir = outputs_dir / "visualizations"
    pose_dir = outputs_dir / "pose_estimation"
    refinement_dir = outputs_dir / "refinement"
    scene_dir = data_dir / "chess"
    return PathsConfig(
        project_dir=project_dir,
        data_dir=data_dir,
        outputs_dir=outputs_dir,
        checkpoints_dir=checkpoints_dir,
        visualizations_dir=visualizations_dir,
        pose_dir=pose_dir,
        refinement_dir=refinement_dir,
        scene_dir=scene_dir,
    )


def ensure_directories(paths: PathsConfig) -> None:
    for directory in [
        paths.project_dir,
        paths.data_dir,
        paths.outputs_dir,
        paths.checkpoints_dir,
        paths.visualizations_dir,
        paths.pose_dir,
        paths.refinement_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
