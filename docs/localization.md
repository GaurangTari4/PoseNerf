# Localization

Inference uses query RGB, fixed intrinsics, training RGB descriptors, and training-only geometry. Query depth and GT pose are excluded. D0 uses measured training depth; D1/D2 use frozen E3/E6 rendered depth with identical SIFT matching and RANSAC-PnP. Official held-out results keep D0 as the preferred method.
