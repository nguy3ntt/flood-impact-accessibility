# Evaluation protocol

## Mapping
Keep an official-split result for comparability where valid, plus a stricter event-group-held-out protocol. Do not describe a chip split as event-generalisation. Freeze manifests and inspect adjacent/overlapping tile leakage. Reserve separate training, selection/calibration and final test groups. If too few events exist for all goals, use nested grouped validation or limit claims and document the compromise.

Primary metrics: water IoU and F1 over common valid pixels; event-macro aggregation. Secondary: precision/recall, per-event errors, class-balanced summaries, calibration/Brier score where appropriate, valid/retained coverage, inference time and memory. Pixel-pooled metrics are secondary because large events dominate. Bootstrap by event/group, not individual pixels; intervals from few events can be unstable.

Thresholds, normalisation, early stopping, calibrators and model selection use training/validation only. Every method sees identical eligible labels/folds; report unavailable-sensor subsets separately. Compare radar thresholding, optical indices where appropriate, compact sensor-specific models, fusion and conditional optical foundation adaptation.

Ablations: sensor contribution, optical quality/missingness, label budget, pretrained versus random encoder, permanent-water handling, event transfer and compute. Separate real cloud/missingness observations from injected stress tests. Inspect model errors rather than only average metrics.

## Uncertainty
Report empirical reliability by event/domain/quality, confidence versus error and abstention coverage. Fit on held-out calibration groups. Correlated pixels and domain shift limit exchangeability assumptions; no automatic conformal guarantee. Preserve nodata and distinguish model uncertainty from missing evidence.

Use spatially coherent masks/ensembles and declared disruption-rule sensitivity. Unless validated probability models exist, report scenario ranges and sensitivity rather than probabilistic access guarantees.

## Network and accessibility
Validate baseline topology, direction, estimated costs, snapping, disconnected components and boundary buffers before flood scenarios. Synthetic invariants: no-disruption equals baseline; removing edges cannot improve shortest cost on the same graph; added nonnegative edges cannot worsen it; missing routes are null; unknown policy affects only the documented edges.

Report baseline-unreachable and newly-disconnected origins separately; access-cost changes among paired reachable origins; origin ranking/scenario stability; unsupported-edge/facility coverage. Population exposure is secondary only with dated population data. Validate exposure or closure against independent data if available; without it, network consequences remain scenarios, not measured real closures.

## Reproducibility and promotion
Store code/environment revision, source/split/config hashes, seed, checkpoint, full prediction references, exclusions, metrics and compute for each run. Promote models only for held-out benefit relevant to the application or a justified reliability/compute advantage. Record negative findings. Preregister primary endpoints before the full advanced study.
