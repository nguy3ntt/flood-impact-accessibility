"""Read-only local analyst API over one validated, immutable M7 bundle."""

from __future__ import annotations

from functools import lru_cache
import gzip
import json
import mimetypes
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .analysis_bundle import load_bundle


def create_app(bundle_root: Path, *, static_root: Path | None = None) -> FastAPI:
    about, roads, evidence, scenarios = load_bundle(bundle_root)
    bundle_root = bundle_root.resolve()
    closure_files = {(row["scenario_id"], row["policy"], row["bridge_policy"]): row["path"]
                     for row in about["closure_files"]}
    app = FastAPI(title="Flood Access Analyst", version="1.0.0", docs_url=None, redoc_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    def selected(scenario_id: str, policy: str, bridge_policy: str) -> dict:
        row = scenarios.get((scenario_id, policy, bridge_policy))
        if row is None:
            raise HTTPException(status_code=404, detail="Unknown scenario or assumption combination")
        return row

    @lru_cache(maxsize=8)
    def closure_ids(scenario_id: str, policy: str, bridge_policy: str) -> tuple[str, ...]:
        row = selected(scenario_id, policy, bridge_policy)
        path = bundle_root / closure_files[(scenario_id, policy, bridge_policy)]
        with gzip.open(path, "rt", encoding="utf-8") as source:
            closed = json.load(source)
        if len(closed) != row["closure"]["closed_physical_segments"] or \
           len(closed) != len(set(closed)) or not set(closed) <= roads.keys():
            raise RuntimeError("Bundle closure IDs differ from source result")
        return tuple(closed)

    @app.get("/api/v1/about")
    def get_about():
        return about

    @app.get("/api/v1/roads")
    def get_roads():
        return {"schema_version": about["schema_version"], "columns": ["segment_id", "lon1", "lat1",
                "lon2", "lat2", "road_class", "length_m", "bridge", "tunnel"],
                "roads": list(roads.values()), "evidence_segment_ids": list(evidence)}

    @app.get("/api/v1/roads/{segment_id}")
    def get_road(segment_id: str):
        road = roads.get(segment_id)
        if road is None:
            raise HTTPException(status_code=404, detail="Unknown road segment ID")
        observed = evidence.get(segment_id)
        return {"segment_id": segment_id, "geometry": {"lon1": road[1], "lat1": road[2],
                 "lon2": road[3], "lat2": road[4]}, "road_class": road[5],
                "length_m": road[6], "bridge": road[7], "tunnel": road[8],
                "evidence_state": "within_image" if observed else "outside_image_unknown",
                "evidence": observed,
                "interpretation": "Road-water overlap is exposure evidence; passability is unknown."}

    @app.get("/api/v1/scenarios/{scenario_id}")
    def get_scenario(scenario_id: str, policy: str = Query(...), bridge_policy: str = Query(...)):
        row = selected(scenario_id, policy, bridge_policy)
        return {**row, "closed_segment_ids": closure_ids(scenario_id, policy, bridge_policy),
                "closed_segment_interpretation": "Simulated removal under the selected assumption, not observed closure"}

    @app.get("/api/v1/previews/{scenario_id}.png")
    def get_preview(scenario_id: str):
        name = about["preview_by_scenario"].get(scenario_id)
        if name is None:
            raise HTTPException(status_code=404, detail="Unknown mapping case ID")
        return FileResponse(bundle_root / name, media_type="image/png",
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/export")
    def export_selected(scenario_id: str = Query(...), policy: str = Query(...),
                        bridge_policy: str = Query(...)):
        row = selected(scenario_id, policy, bridge_policy)
        payload = {"schema_version": "m7_selected_scenario_export_v1", "mode": about["mode"],
                   "case_id": about["case_id"], "bundle_run_id": about["run_id"],
                   "m6_source_run_id": about["source_run_id"],
                   "m6_source_manifest_sha256": about["source_manifest_sha256"],
                   "dates": about["dates"], "coverage": about["coverage"],
                   "baseline": about["baseline"], "selected_scenario": row,
                   "interpretation": about["interpretation"], "attribution": about["attribution"]}
        return JSONResponse(payload, headers={"Content-Disposition":
                            f'attachment; filename="{about["run_id"]}-selected-scenario.json"',
                            "Cache-Control": "no-store"})

    ui_root = static_root or Path(__file__).resolve().parents[2] / "web" / "static"
    if not (ui_root / "index.html").is_file() or not (ui_root / "app.js").is_file():
        raise FileNotFoundError("Built M7 interface missing; run npm run build in web/")
    mimetypes.add_type("font/woff2", ".woff2")
    app.mount("/static", StaticFiles(directory=ui_root), name="static")

    @app.get("/")
    def home():
        return RedirectResponse("/static/index.html", status_code=307)

    return app
