# Reproducibility

The starter can run entirely on synthetic fixtures after installation; tests also run with PYTHONPATH=src. Its package has no runtime dependencies. Resolve and lock scientific dependencies during M0/M1 after hardware and wheel-compatibility checks; no pretrained weights are included.

Every real experiment needs immutable source manifests, group-split hashes, preprocessing/config revision, code revision, environment lock, seeds, exact model revision and weights hash, input/output hashes and a complete run status. Keep raw data and large artefacts outside Git, with documented logical paths and regeneration commands.

Distinguish a cached-result rebuild from fresh model training. A clean clone must run the synthetic smoke mode, while real-data results may need licensed downloads, credentials, time and GPU resources. Report these requirements explicitly. Do not promise byte-identical neural training across hardware; define tolerances and record deterministic settings.

Until M8, the public repository is work in progress and has no real-data performance claims. CI initially verifies only the actual lightweight code. Extend checks as raster/model/API functionality is implemented.
