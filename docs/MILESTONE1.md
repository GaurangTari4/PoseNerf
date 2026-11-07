# Milestone 1: Scientific Correctness and Repository Reliability

## Status

- [x] Coarse and fine NeRF outputs are both supervised during training.
- [x] Validation frames are disjoint from training frames; official test frames are not used for checkpoint selection.
- [x] RGB localization uses training-view RGB retrieval and RGB-only photometric pose refinement.
- [x] Test pose and depth are evaluation-only in the localization path.
- [x] Refinement trains only on training views and selects checkpoints on validation frames.
- [x] Checkpoint loading requires `weights_only=True` and strict model-state matching.
- [x] Regression tests cover coarse gradients, split disjointness, and the identity-camera ray convention.
- [ ] Validate intrinsics, depth scale, and pose convention against a downloaded Chess sample.
- [x] Run CPU and CUDA synthetic tests in a project environment with PyTorch installed.
- [ ] Benchmark rendering/localization quality on the official test set.

## Evaluation protocol

The official 7-Scenes `TrainSplit.txt` frames are deterministically partitioned into training and validation subsets. The official `TestSplit.txt` frames are reserved for final rendering and localization evaluation. Ground-truth test poses are loaded only to calculate errors after inference. Test RGB-D values never enter pose initialization or pose optimization.

## Known limitations

The initializer is global RGB nearest-view retrieval, which is an intentionally simple, dependency-free baseline. It can fail for visually similar or distant views. No claim of iNeRF/BARF reproduction is made. The project still loads selected tensors and ray pools on the selected device; memory-streaming work remains a later milestone.
