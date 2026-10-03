"""Build reviewed event-only M8 tables and vector chart from a verified local run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from stage_m8_final import write_json
from verify_m8_final import verify


OUT = Path("docs/results")
METHOD_LABELS = {
    "radar_vh": "Radar VH threshold", "ndwi": "NDWI", "mndwi": "MNDWI",
    "logistic": "Logistic baseline", "radar_unet": "Radar U-Net",
    "optical_unet": "Optical U-Net", "m4_4107_radar": "Fusion radar, seed 4107",
    "m4_4107_optical": "Fusion optical, seed 4107",
    "m4_4107_early": "Early fusion, seed 4107",
    "m4_4107_robust_early": "Dropout fusion, seed 4107",
    "m4_4107_late": "Late fusion, seed 4107",
    "m4_4108_radar": "Fusion radar, seed 4108",
    "m4_4108_optical": "Fusion optical, seed 4108",
    "m4_4108_early": "Early fusion, seed 4108",
    "m4_4108_robust_early": "Dropout fusion, seed 4108",
    "m4_4108_late": "Late fusion, seed 4108",
}


def export(run_id: str) -> dict:
    audit = verify(run_id)
    root = Path("runs") / run_id
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for name, event in summary["events"].items():
        rows.append({"event": name, "chips": event["chips"],
                     "common_eligible_pixels": event["eligible_pixels"],
                     "observed_water_fraction": event["water_prevalence"],
                     "methods": {method: {"iou": result["iou"], "f1": result["f1"]}
                                 for method, result in event["methods"].items()},
                     "calibration": {key: {metric: event["reliability"][key][metric]
                                           for metric in ("brier", "ece", "prevalence")}
                                     for key in ("calibrated", "unit_scaled_ndwi", "calibration_prevalence")},
                     "quality_and_ambiguity_retention": {
                         "retained_fraction": event["retained"]["fraction"],
                         "withheld_observed_water_pixels": event["retained"]["withheld_water"],
                         "retained_brier": event["retained"]["reliability"]["brier"]}})
    result = {"schema_version": "m8_public_event_aggregate_v1", "source_run_id": run_id,
              "source_manifest_sha256": audit["manifest_sha256"],
              "benchmark": "Sen1Floods11 v1.1 hand-labelled observed-water test chips",
              "attribution": "Cloud to Street; Bonafilia et al. (2020), Sen1Floods11.",
              "split": "Frozen previously untouched Nigeria 2018-09-21 and Somalia 2018-05-07 events; 44 chips.",
              "unit": "Event: confusion counts pooled within event, then two event scores averaged equally.",
              "support": "Label known, both sensors valid, NDWI and MNDWI denominators defined; identical support for all listed classifiers.",
              "methods": METHOD_LABELS, "events": rows,
              "event_macro": {name: {"iou": metric["iou"], "f1": metric["f1"]}
                              for name, metric in summary["event_macro"].items()},
              "limitations": ["Water label is not an independently verified event-flood, depth or road-closure label.",
                              "Two events do not establish reliable geographic transfer or a narrow confidence interval.",
                              "Calibration is empirical; event shift gives no coverage guarantee.",
                              "No final labels were used to fit, select, tune or promote a model."]}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "final_water_benchmark.json", result)
    chosen = ["radar_vh", "ndwi", "mndwi", "logistic", "radar_unet", "optical_unet",
              "m4_4107_optical", "m4_4108_optical", "m4_4107_early", "m4_4108_early",
              "m4_4107_robust_early", "m4_4108_robust_early", "m4_4107_late", "m4_4108_late"]
    width, height = 1100, 1030
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">Final observed-water IoU by frozen method</title>',
           '<desc id="desc">Two-event macro IoU on Nigeria and Somalia final-test chips. MNDWI is 0.777; optical U-Net seed results vary.</desc>',
           '<rect width="1100" height="1030" fill="#f4f6f3"/>',
           '<text x="42" y="48" font-family="sans-serif" font-size="26" font-weight="700" fill="#17383d">Frozen final-test water benchmark</text>',
           '<text x="42" y="76" font-family="sans-serif" font-size="15" fill="#425e63">44 chips · 2 events · equal event weighting · identical eligible pixels</text>']
    for index, name in enumerate(chosen):
        y = 117 + 58 * index
        score = result["event_macro"][name]["iou"]
        colour = "#127d80" if name == "mndwi" else "#607e82" if name.startswith("m4_") else "#9bb8b6"
        svg.extend([f'<text x="42" y="{y+19}" font-family="sans-serif" font-size="15" fill="#17383d">{escape(METHOD_LABELS[name])}</text>',
                    f'<rect x="330" y="{y}" width="650" height="26" fill="#dce6e2"/>',
                    f'<rect x="330" y="{y}" width="{650*score:.1f}" height="26" fill="{colour}"/>',
                    f'<text x="990" y="{y+19}" font-family="monospace" font-size="16" fill="#17383d">{score:.3f}</text>'])
    svg.extend(['<text x="42" y="980" font-family="sans-serif" font-size="14" fill="#425e63">Observed-water label only; no closure or passability validation. Two events limit generalisation.</text>',
                f'<text x="42" y="1005" font-family="sans-serif" font-size="13" fill="#425e63">Source: Cloud to Street / Bonafilia et al. (2020), Sen1Floods11; local run {escape(run_id)}.</text>', '</svg>'])
    (OUT / "final_water_iou.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    return {"source_run_id": run_id, "events": len(rows), "public_outputs": [str(OUT / "final_water_benchmark.json"),
                                                                     str(OUT / "final_water_iou.svg")]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(export(args.run_id), indent=2))
