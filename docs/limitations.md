# Limitations

- Training uses known poses; joint unknown-pose NeRF training is not evaluated.
- E3/E6 weights are local artifacts, not Git binaries.
- NeRF-depth PnP underperforms measured-depth PnP on held-out views.
- iNeRF-inspired refinement can lower RGB error while worsening pose error.
- Studio browser automation is skipped where Playwright is unavailable.
