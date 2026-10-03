# Local analyst application (M7)

The M7 application serves one **immutable, validated local bundle** at a time. The real bundle is derived from the verified Spain M6 scenario run; the synthetic bundle is generated from invented roads, coordinates and access values. Neither mode is an emergency navigation service. The server listens on `127.0.0.1` only and ordinary requests read precomputed files; they do not train models or run raster inference.

## Start the synthetic clean-clone demo

From the repository root with Python 3.12 and Node.js 24:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install -r requirements/m7-serve.lock
cd web
npm ci
npm run build
cd ..
.\.venv\Scripts\python.exe scripts/build_m7_bundle.py --synthetic
.\.venv\Scripts\python.exe scripts/verify_m7_bundle.py <printed-M7S-ID>
.\.venv\Scripts\python.exe scripts/serve_m7.py <printed-M7S-ID>
```

Open `http://127.0.0.1:8765/`. The header says **SYNTHETIC · INVENTED**. The sample has two origins, one hospital and four invented roads. Selecting `demo water` removes one invented road segment from the simulated graph and disconnects one invented origin. Selecting `demo quiet` removes none. These values are test fixtures, never field results. Stop the local server with Ctrl+C.

## Open the bounded Spain case

The real build additionally needs the ignored verified M2, M5 and M6 runs, source files and `requirements/m2-pipeline.lock`. The local authoritative M6 input is `M6-7b030cf0d570`.

```powershell
.\.venv\Scripts\python.exe scripts/build_m7_bundle.py --m6-run-id M6-7b030cf0d570
.\.venv\Scripts\python.exe scripts/verify_m7_bundle.py <printed-M7-ID>
.\.venv\Scripts\python.exe scripts/serve_m7.py <printed-M7-ID>
```

The builder first replays M6 verification, then derives a versioned bundle of road geometry, road evidence, exact M6 scenario rows, per-scenario assumed-removal IDs and georeferenced image previews. It checks each scenario's closed-segment counts and reasons before writing a complete manifest. The verifier rechecks every output hash, source linkage and all 92 scenario rows; its deep mode also replays M6 and all closure IDs. A completed run ID is immutable. A code/input change yields a different ID; a failed or earlier run stays under ignored `runs/` for audit.

## Interface and manual checks

The analysis console reads from top to bottom: case context, geographic evidence, selected graph result, a full-width layer/provenance band, origin/facility ledger, and interpretation limits. The numbered control desk sets the water case, road evidence rule and bridge treatment; the evidence band explains map colours and source dates. The condensed headings and monospaced identifiers use locally bundled IBM Plex fonts (OFL 1.1 licences in `web/static/fonts/`); no font service or basemap request is needed.

1. Confirm the header says **LOCAL RESEARCH CASE**. In **Data provenance**, compare road, radar, optical, independent trace and hospital-list dates. The hospital list is from 2026; its 2019 operating status is unknown.
2. Choose **M3 fixed NDWI reference**, **All source-supported overlap**, **Bridge overlap exempt**. The result should show 2 newly disconnected of 11 baseline-reachable origins and 252 assumed-removed physical segments. Select **Quality-retained overlap only** and check the count changes to 1. Select **M5 score ≥ 0.50** with source-supported overlap and check it changes to 0. These are different assumptions, not measured closure rates.
3. Click **Image area**, toggle the water layer in **Read the map**, then use the legend. Teal marks retained water candidates, amber indicates withheld image evidence, gray roads lie outside the image and red roads are assumed removed under the selected graph case. Cloud status is unknown.
4. Open **Inspect road evidence**. Check exposed segment `osm:221400844:7`, bridge case `osm:54618235:0` and outside-image segment `osm:100014568:0`. The outside segment must say water status **unknown**, never dry. The bridge tag does not prove passability.
5. Compare baseline and selected costs in the origin table, including `INE2019_section_0306201001`. Sort by origin or costs. The designed-member count is a sensitivity count, not a probability. Facility positions are snapped road nodes and one 2026 hospital candidate lies outside the study box.
6. Export the selected analysis JSON. It contains source dates, coverage, baseline and selected scenario results, assumptions and attribution. It excludes raw imagery, labels, all-road geometry and weights. Repeat at desktop, tablet and narrow phone widths: controls, provenance, map actions and ledger should remain usable without horizontal page overflow (the ledger table scrolls within its own frame on phones).

## API contract

All endpoints are read-only and versioned under `/api/v1`:

| Endpoint | Response |
|---|---|
| `GET /api/v1/about` | Bundle schema/mode, dates, coverage, origins, hospitals, scenario options and interpretation limits. |
| `GET /api/v1/roads` | Local graph segment geometry and IDs for the map. This is local derived data, not a public source download. |
| `GET /api/v1/roads/{segment_id}` | One segment's source support, unknown fraction, retained area, mapping overlap and grade tags; 404 for an unknown ID. |
| `GET /api/v1/scenarios/{scenario_id}?policy=...&bridge_policy=...` | Exact M6 scenario summary/origin rows and assumed-removed segment IDs; 404 for an unavailable combination, 422 for omitted required parameters. |
| `GET /api/v1/previews/{scenario_id}.png` | Local image preview for a declared mapping case; 404 for an unknown ID. |
| `GET /api/v1/export?scenario_id=...&policy=...&bridge_policy=...` | Bounded JSON export for one selected scenario; invalid options fail clearly. |

The bundle schema is `m7_analyst_bundle_v1`. `manifest.json` and `status.json` bind output hashes, code hashes, source M6 manifest hash and completion. `load_bundle` rejects changed/missing/extra outputs, invalid road geometry, unsupported evidence/scenario joins and inconsistent origin support before the API starts. The server uses an exact bundle ID, never an arbitrary client file path. Public Git contains source, locks and the reviewed compiled UI, but not the generated bundles or pixel data.

The map shows only one small image tile over a much wider graph. Unknown off-image roads must remain unknown. Road-water overlap is exposure evidence; deterministic edge removal is a graph assumption. Estimated graph-node costs omit origin/facility entrance connectors, true traffic, water depth and verified passability. No scenario path is a recommended public route.
