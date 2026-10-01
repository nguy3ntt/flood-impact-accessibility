# Architecture and implementation boundaries

## Offline research pipeline
Acquisition/catalog → raster/vector validation → canonical aligned data → fixed experiment folds → baseline/advanced mapping → calibration/quality outputs → road exposure → declared disruption scenarios → accessibility → versioned serving bundle.

Responsibilities should be separate modules: acquisition, catalog/contracts, raster preprocessing, splits, datasets/model training, inference/calibration, vector/graph construction, exposure, scenario generation, accessibility, evaluation and bundle export. Introduce real modules when needed, not empty placeholder implementations.

Store raw files immutably; canonical tables in Parquet/GeoParquet; raster results in appropriate georeferenced formats; run metadata in JSON. Use checksums and logical IDs. A local file-based pipeline is sufficient initially. A database, queue or cloud platform requires a concrete need.

## Serving plane, planned for M7
FastAPI reads validated precomputed bundles. A TypeScript map app displays imagery/result tiles, graph evidence, facilities, origins and scenario comparisons. Expensive training and raster inference do not run on ordinary map requests. The API validates area/event/scenario IDs, provides bounded exports and exposes model/data dates and unknown reasons. Documented endpoint contracts come from the actual implemented engine.

Proposed interface: map-first workspace, layer/coverage legend, scenario comparison panel, selected-road evidence drawer and sortable origin-service table. Include keyboard access, readable colour scales, dark/light themes only if useful, and responsive checks. Design for this analytical task rather than copying layouts from prior apps.

## Included implementation
Only the standard-library synthetic graph engine, status/demo CLI, configuration skeleton, manifest template, tests and CI are implemented. There is no server, map UI, downloader or trained network yet. The synthetic engine illustrates graph reachability under assumed edge states; geospatial topology/weights are not solved by this example.
