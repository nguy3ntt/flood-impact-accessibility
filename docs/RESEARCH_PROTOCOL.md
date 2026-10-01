# Research protocol

The `event_holdout_v1` split is fixed before model experiments. Hypotheses and numeric acceptance targets remain draft until initial simple baselines, before advanced experiments. Evaluation details and data semantics are in [EVALUATION.md](EVALUATION.md), [SCOPE.md](SCOPE.md), [DATA_CONTRACT.md](DATA_CONTRACT.md), [DATASETS.md](DATASETS.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Research objective and units

Evaluate whether radar/optical mapping and evidence-quality controls improve retrospective observed-water mapping across held-out events, then examine how mapping uncertainty and explicit road-disruption assumptions affect simulated hospital accessibility. The mapping evaluation unit is the event; pixels are dependent observations within it. The accessibility unit is an origin in one geographically coherent, dated case study. An exposed road is not a verified closure.

## Primary questions and planned comparisons

1. **Sensor value:** compare eligible radar thresholding and optical index methods with compact radar-only, optical-only, early-fusion and late-fusion models on common valid pixels and event-held-out folds. Test observed and injected optical degradation separately.
2. **Evidence quality:** evaluate calibration, error and retained coverage by event, sensor availability and image quality. Missing evidence remains unknown. Test whether quality filtering reduces unsupported conclusions while retaining useful coverage.
3. **Accessibility stability:** compare baseline graph access with declared edge-disruption scenarios built from spatially coherent water alternatives. Report baseline-unreachable and newly-disconnected origins separately, paired cost changes, ranking stability and sensitivity to bridge and unknown-edge rules.
4. **Conditional transfer:** compare a compatible pretrained optical encoder with a compact model and random encoder across fixed label budgets only if M1 establishes appropriate inputs, terms and compute capacity. This branch does not replace the core fusion comparison.

## Design and decision rules

Use the official Sen1Floods11 chip split only for benchmark comparability: M1 verified that train, validation and test share the same ten event groups. The [event-group split](../configs/event_split_v1.json) is frozen for primary transfer analysis after M2 checked aliases and all 446 STAC footprints: 6 training, 2 validation and 2 untouched final-test events, with Spain reserved as an exploratory case-study event. M1 had already inspected a Spain pilot label, so Spain is not a pristine final benchmark test. Do not use Nigeria/Somalia final-test labels for thresholds, preprocessing choices, stopping, calibration or model selection. Two final groups give limited precision; show individual event results and use grouped development folds for stability checks rather than pixel-level confidence claims.

Primary mapping endpoints are water IoU and F1, aggregated by event over the same eligible valid pixels. Report precision, recall, error cases, reliability, coverage and compute as supporting outcomes. Summarise uncertainty by event/group, not independent pixels. Promote a more complex model only for relevant held-out benefit or a justified reliability/compute advantage; retain negative results.

Network results are conditional on graph quality, source timing and scenario policy. Report scenario ranges and sensitivity unless independently justified probabilities exist. No pixel calibration score is interpreted as a road-closure probability. No route is presented as public travel advice.

## Feasibility gates before freezing

M1 verified live source access, real raster/label bytes, band descriptions, nodata codes, one-day sensor offsets in two pilot events, split leakage, source-rights uncertainty and limited storage. M2 established the event split and canonical three-chip pilot; optical cloud status remains unknown and the Spain road graph remains unbuilt. Spain 2019 is the conditional first exposure case, while Brisbane 2022 peak fusion is no-go with current evidence. Numeric acceptance targets will be set after initial M3 baseline measurements, before advanced full experiments.
