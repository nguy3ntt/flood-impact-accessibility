# Architecture and implementation boundaries

## Offline research pipeline
Acquisition/catalog → raster/vector validation → canonical aligned data → fixed experiment folds → baseline/advanced mapping → calibration/quality outputs → road exposure → declared disruption scenarios → accessibility → versioned serving bundle.

Responsibilities should be separate modules: acquisition, catalog/contracts, raster preprocessing, splits, datasets/model training, inference/calibration, vector/graph construction, exposure, scenario generation, accessibility, evaluation and bundle export. Introduce real modules when needed, not empty placeholder implementations.

Store raw files immutably; canonical and graph tables in validated, compressed local files; raster results in georeferenced formats; run metadata in JSON. Use checksums and logical IDs. A local file-based pipeline is sufficient initially. Parquet/GeoParquet, a database, queue or cloud platform require a concrete need.

## Serving plane (M7)
FastAPI reads one hash-validated precomputed bundle selected when the local service starts. A compiled TypeScript canvas map displays case-image previews, graph evidence, facilities, origins and scenario comparisons without external map tiles. Expensive training and raster inference do not run on ordinary map requests. The API validates exact road/scenario/assumption IDs, provides bounded selected-scenario exports and exposes source dates and unknown reasons. The implemented endpoint contract and manual workflow are in [Analyst app](ANALYST_APP.md).

The interface has a map-first workspace, layer/coverage legend, scenario comparison panel, selected-road evidence drawer and sortable origin-service table. Road search and tables provide keyboard alternatives to canvas selection. The bundle keeps raw imagery/labels and source extracts outside Git; local previews and derived graph geometry remain in ignored `runs/`.

## Included implementation
M2 validates raster/vector inputs and event groups; M3/M4 run mapping and fusion experiments; M5 fits an empirical water score and writes quality/scenario rasters. M6 acquires a capped historical Spain OSM subset, builds directed edges only through shared OSM node IDs, samples label-independent case imagery along road geometry, and compares static graph costs under declared edge-removal policies. Graph nodes/edges and road evidence are compressed local tables, accompanied by georeferenced rasters, summaries, hashes and QA panels. M7 verifies M6 before building a source-linked serving bundle. One tile has water evidence; off-tile roads remain unknown. The CLI and an invented synthetic demo remain available from a clean clone.
