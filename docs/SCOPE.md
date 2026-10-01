# Scope and success criteria

## Core research product
A reproducible historical-event analysis linking evaluated water mapping to scenario-based hospital accessibility. Primary outputs are raster evidence, road exposure records and origin-service comparison tables. Analytical paths and costs illustrate assumptions; they do not instruct travellers.

## Output semantics
`water_probability`: model score for observed water under the training label definition. `event_inundation`: only where independent baseline/reference evidence supports the distinction. `road_exposure`: mapping/geometry overlap with supporting pixel/time information. `scenario_disrupted`: edge disabled by an explicit assumption. `unknown`: insufficient image or network evidence. None is interchangeable with verified closure.

## Minimum defensible completion
- A real labelled mapping benchmark with per-event held-out results and inspected errors.
- A geographically coherent case study with dated sources, documented validation status and graph-quality audit.
- A baseline/advanced comparison and one well-controlled sensor/quality ablation.
- Calibration and uncertainty/unknown analysis without unjustified guarantees.
- Hospital reachability and estimated cost comparison across multiple declared scenarios.
- A reproducible local application and clean-copy synthetic smoke mode.
- Public method/result/limitation documentation with provenance and attribution.

Numeric acceptance targets should be preregistered after feasibility and baseline measurements, before full advanced experiments. Do not promise an arbitrary IoU or improvement. A negative model comparison can be successful research if rigorous and honestly presented.

## Boundaries
One event-area demonstrator, one service category and a manageable graph are sufficient. No forecast of future flood hazard, water depth/velocity, live closures, evacuation optimisation, hospital capacity, patient outcomes or safety certification is implied. Population weighting and mitigation recommendations require additional data and validation.

## Default analyst workflow
Select historical event → inspect imagery and coverage → inspect mapping evidence → select explicit disruption assumptions → compare baseline and scenario access → investigate ambiguous links → export reproducible analysis. Every screen must retain dates, scenario labels and unknown-state meaning.
