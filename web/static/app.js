"use strict";
function element(id) {
    const value = document.getElementById(id);
    if (!value)
        throw new Error(`Missing interface element: ${id}`);
    return value;
}
const canvas = element("case-map");
const ctx = canvas.getContext("2d");
if (!ctx)
    throw new Error("Canvas 2D is unavailable");
const drawing = ctx;
const scenarioSelect = element("scenario-select");
const policySelect = element("policy-select");
const bridgeSelect = element("bridge-select");
const drawer = element("evidence-drawer");
const roadDetail = element("road-detail");
let about;
let roads = [];
let roadIds = new Set();
let evidenceIds = new Set();
let selected = null;
let closedIds = new Set();
let selectedRoad = null;
let preview = null;
let currentPreviewId = "";
let sortAscending = true;
let sortBy = "origin";
const map = { centerX: 0, centerY: 0, scale: 1, cosLat: 1, width: 1, height: 1 };
let pointerStart = null;
async function json(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
        let message = `${response.status} ${response.statusText}`;
        try {
            const body = await response.json();
            message = body.detail ?? message;
        }
        catch { /* keep HTTP text */ }
        throw new Error(message);
    }
    return await response.json();
}
function setText(id, value) { element(id).textContent = value; }
function showError(error) {
    const banner = element("error-banner");
    banner.hidden = false;
    banner.textContent = `Unable to load the local analysis: ${error instanceof Error ? error.message : String(error)}`;
}
function number(value, digits = 0) {
    return value == null ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: digits });
}
function cost(value) {
    return value == null ? "No graph path" : `${number(value / 60, 1)} min`;
}
function labelForScenario(id) {
    if (id === "m3_ndwi_reference")
        return "M3 fixed NDWI reference";
    if (id === "threshold_0.50")
        return "M5 score ≥ 0.50";
    if (id.startsWith("threshold_"))
        return `M5 score ≥ ${id.slice(10)}`;
    if (id.startsWith("m3_ndwi_threshold_"))
        return `M3 NDWI ${id.slice(18)}`;
    if (id.startsWith("m3_shared_offset_seed_"))
        return `M3 spatial case ${id.slice(22)}`;
    if (id.startsWith("offset_seed_"))
        return `M5 spatial case ${id.slice(12)}`;
    return id.replaceAll("_", " ");
}
const policyLabels = {
    supported_overlap: "All source-supported overlap",
    retained_overlap: "Quality-retained overlap only",
    supported_plus_within_tile_unknown: "Overlap + withheld image area",
    supported_plus_all_unknown: "Overlap + all unknown roads (stress only)",
};
const bridgeLabels = {
    bridge_exempt: "Bridge overlap exempt",
    bridge_candidate: "Bridge overlap may remove",
};
function options(select, entries, preferred) {
    select.replaceChildren(...entries.map(([value, label]) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        return option;
    }));
    if (preferred && entries.some(([value]) => value === preferred))
        select.value = preferred;
}
function distinct(values) { return [...new Set(values)]; }
function updateAvailable(changed = "scenario") {
    const selectedScenario = scenarioSelect.value;
    const matching = about.scenario_options.filter(row => row.scenario_id === selectedScenario);
    const priorPolicy = policySelect.value;
    options(policySelect, distinct(matching.map(row => row.policy)).map(value => [value, policyLabels[value] ?? value]), changed === "scenario" ? "supported_overlap" : priorPolicy);
    const priorBridge = bridgeSelect.value;
    const available = matching.filter(row => row.policy === policySelect.value);
    options(bridgeSelect, distinct(available.map(row => row.bridge_policy)).map(value => [value, bridgeLabels[value] ?? value]), changed === "bridge" ? priorBridge : "bridge_exempt");
    void loadScenario().catch(showError);
}
async function loadScenario() {
    const id = scenarioSelect.value;
    const policy = policySelect.value;
    const bridge = bridgeSelect.value;
    if (!id || !policy || !bridge)
        return;
    const query = new URLSearchParams({ policy, bridge_policy: bridge });
    const row = await json(`/api/v1/scenarios/${encodeURIComponent(id)}?${query}`);
    if (scenarioSelect.value !== id || policySelect.value !== policy || bridgeSelect.value !== bridge)
        return;
    selected = row;
    closedIds = new Set(row.closed_segment_ids);
    setText("metric-disconnected", number(row.summary.newly_disconnected));
    setText("metric-closed", number(row.closure.closed_physical_segments));
    setText("metric-paired", `${row.summary.paired_reachable}/${row.summary.origins}`);
    setText("metric-baseline", `${row.summary.origins - row.summary.baseline_unreachable}/${row.summary.origins} baseline reachable · paired cost increase ${number(row.summary.mean_paired_cost_increase_seconds, 1)} s`);
    setText("scenario-assumption", policy === "supported_plus_all_unknown" ?
        "Extreme coverage stress: every unknown road is assumed removed. This is not an observed disruption." :
        "A qualifying road segment is removed from the simulated graph. Overlap does not establish closure or passability.");
    const url = new URL("/api/v1/export", window.location.origin);
    url.search = new URLSearchParams({ scenario_id: id, policy, bridge_policy: bridge }).toString();
    element("export-link").href = url.toString();
    renderOrigins();
    if (selectedRoad)
        void inspectRoad(selectedRoad, false);
    if (currentPreviewId !== id) {
        currentPreviewId = id;
        const image = new Image();
        image.onload = () => { if (currentPreviewId === id) {
            preview = image;
            draw();
        } };
        image.onerror = () => showError(`Preview missing for ${id}`);
        image.src = `/api/v1/previews/${encodeURIComponent(id)}.png`;
    }
    setText("map-status", `${labelForScenario(id)} · ${policyLabels[policy] ?? policy} · simulated graph`);
    draw();
}
function world(lon, lat) { return [lon * map.cosLat, -lat]; }
function project(lon, lat) {
    const [x, y] = world(lon, lat);
    return [map.width / 2 + (x - map.centerX) * map.scale,
        map.height / 2 + (y - map.centerY) * map.scale];
}
function fit(bounds) {
    const [south, west, north, east] = bounds;
    map.cosLat = Math.cos(((south + north) / 2) * Math.PI / 180);
    const [minX, maxY] = world(west, south);
    const [maxX, minY] = world(east, north);
    map.centerX = (minX + maxX) / 2;
    map.centerY = (minY + maxY) / 2;
    map.scale = Math.min((map.width - 65) / Math.max(maxX - minX, 0.00001), (map.height - 65) / Math.max(maxY - minY, 0.00001));
    draw();
}
function resize() {
    const box = canvas.getBoundingClientRect();
    map.width = Math.max(1, box.width);
    map.height = Math.max(1, box.height);
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(map.width * ratio);
    canvas.height = Math.round(map.height * ratio);
    drawing.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (about && map.scale === 1)
        fit(about.study_bounds);
    else
        draw();
}
function pathFor(roadsToDraw) {
    const path = new Path2D();
    for (const row of roadsToDraw) {
        const [x1, y1] = project(row[1], row[2]);
        const [x2, y2] = project(row[3], row[4]);
        if (Math.max(x1, x2) < -5 || Math.min(x1, x2) > map.width + 5 ||
            Math.max(y1, y2) < -5 || Math.min(y1, y2) > map.height + 5)
            continue;
        path.moveTo(x1, y1);
        path.lineTo(x2, y2);
    }
    return path;
}
function draw() {
    drawing.clearRect(0, 0, map.width, map.height);
    drawing.fillStyle = "#e6edef";
    drawing.fillRect(0, 0, map.width, map.height);
    if (!about)
        return;
    const [south, west, north, east] = about.study_bounds;
    const [x1, y1] = project(west, north);
    const [x2, y2] = project(east, south);
    drawing.fillStyle = "#f4f7f4";
    drawing.fillRect(x1, y1, x2 - x1, y2 - y1);
    drawing.strokeStyle = "#9bb5b8";
    drawing.lineWidth = 1;
    drawing.setLineDash([7, 5]);
    drawing.strokeRect(x1, y1, x2 - x1, y2 - y1);
    drawing.setLineDash([]);
    const [ts, tw, tn, te] = about.tile_bounds;
    const [tx1, ty1] = project(tw, tn);
    const [tx2, ty2] = project(te, ts);
    drawing.fillStyle = "#d1e9e6";
    drawing.fillRect(tx1, ty1, tx2 - tx1, ty2 - ty1);
    if (element("layer-water").checked && preview) {
        drawing.imageSmoothingEnabled = false;
        drawing.drawImage(preview, tx1, ty1, tx2 - tx1, ty2 - ty1);
    }
    drawing.strokeStyle = "#39a3a6";
    drawing.lineWidth = 2;
    drawing.strokeRect(tx1, ty1, tx2 - tx1, ty2 - ty1);
    if (element("layer-roads").checked && roads.length) {
        const unknown = [], covered = [], removed = [];
        for (const row of roads) {
            if (closedIds.has(row[0]))
                removed.push(row);
            else if (evidenceIds.has(row[0]))
                covered.push(row);
            else
                unknown.push(row);
        }
        drawing.strokeStyle = "#b4c4ca";
        drawing.lineWidth = 1;
        drawing.stroke(pathFor(unknown));
        drawing.strokeStyle = "#69a8bd";
        drawing.lineWidth = 1.4;
        drawing.stroke(pathFor(covered));
        drawing.strokeStyle = "#d9635b";
        drawing.lineWidth = 2;
        drawing.stroke(pathFor(removed));
    }
    if (selectedRoad) {
        const selectedGeometry = roads.find(row => row[0] === selectedRoad);
        if (selectedGeometry) {
            drawing.strokeStyle = "#122f40";
            drawing.lineWidth = 5;
            drawing.stroke(pathFor([selectedGeometry]));
        }
    }
    if (element("layer-origins").checked) {
        for (const origin of about.origins) {
            const [x, y] = project(origin.lon, origin.lat);
            drawing.beginPath();
            drawing.arc(x, y, 5, 0, Math.PI * 2);
            drawing.fillStyle = "#fff";
            drawing.fill();
            drawing.lineWidth = 2;
            drawing.strokeStyle = "#294b5b";
            drawing.stroke();
        }
        for (const hospital of about.hospitals.filter(item => item.status === "snapped" && item.lon != null && item.lat != null)) {
            const [x, y] = project(hospital.lon, hospital.lat);
            drawing.beginPath();
            drawing.arc(x, y, 7, 0, Math.PI * 2);
            drawing.fillStyle = "#164c60";
            drawing.fill();
            drawing.lineWidth = 2;
            drawing.strokeStyle = "#fff";
            drawing.stroke();
            drawing.fillStyle = "#fff";
            drawing.font = "bold 9px sans-serif";
            drawing.textAlign = "center";
            drawing.fillText("H", x, y + 3);
        }
    }
    drawing.fillStyle = "#46636b";
    drawing.font = "11px sans-serif";
    drawing.textAlign = "left";
    drawing.fillText(about.mode === "synthetic" ? "INVENTED GEOGRAPHY" : "ORIHUELA / DOLORES · SPAIN", Math.max(10, tx1), Math.max(36, ty1 - 8));
}
function distanceToSegment(x, y, a, b) {
    const vx = b[0] - a[0], vy = b[1] - a[1];
    const t = Math.max(0, Math.min(1, ((x - a[0]) * vx + (y - a[1]) * vy) / (vx * vx + vy * vy || 1)));
    return Math.hypot(x - (a[0] + t * vx), y - (a[1] + t * vy));
}
function pickRoad(x, y) {
    let best = 9;
    let id = null;
    for (const row of roads) {
        const a = project(row[1], row[2]);
        const b = project(row[3], row[4]);
        if (Math.max(a[0], b[0]) < x - best || Math.min(a[0], b[0]) > x + best ||
            Math.max(a[1], b[1]) < y - best || Math.min(a[1], b[1]) > y + best)
            continue;
        const distance = distanceToSegment(x, y, a, b);
        if (distance < best) {
            best = distance;
            id = row[0];
        }
    }
    return id;
}
function mapPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
}
function zoom(factor, x = map.width / 2, y = map.height / 2) {
    const worldX = map.centerX + (x - map.width / 2) / map.scale;
    const worldY = map.centerY + (y - map.height / 2) / map.scale;
    map.scale = Math.min(2000000, Math.max(200, map.scale * factor));
    map.centerX = worldX - (x - map.width / 2) / map.scale;
    map.centerY = worldY - (y - map.height / 2) / map.scale;
    draw();
}
function textCell(text, className = "") {
    const cell = document.createElement("td");
    cell.textContent = text;
    cell.className = className;
    return cell;
}
function renderOrigins() {
    if (!selected)
        return;
    const byId = new Map(selected.origins.map(row => [row.origin_id, row]));
    const originList = [...about.origins].sort((a, b) => {
        const left = sortBy === "origin" ? a.feature_id :
            sortBy === "baseline" ? a.baseline.estimated_cost_seconds : byId.get(a.feature_id)?.estimated_cost_seconds;
        const right = sortBy === "origin" ? b.feature_id :
            sortBy === "baseline" ? b.baseline.estimated_cost_seconds : byId.get(b.feature_id)?.estimated_cost_seconds;
        const order = typeof left === "string" && typeof right === "string" ? left.localeCompare(right) :
            (left ?? Number.POSITIVE_INFINITY) < (right ?? Number.POSITIVE_INFINITY) ? -1 :
                (left ?? Number.POSITIVE_INFINITY) > (right ?? Number.POSITIVE_INFINITY) ? 1 : 0;
        return (sortAscending ? order : -order) || a.feature_id.localeCompare(b.feature_id);
    });
    const body = element("origin-body");
    body.replaceChildren();
    for (const origin of originList) {
        const result = byId.get(origin.feature_id);
        if (!result)
            continue;
        const tr = document.createElement("tr");
        const name = document.createElement("td");
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = origin.feature_id.replace("INE2019_section_", "Section ");
        button.title = origin.feature_id;
        button.addEventListener("click", () => {
            const [x, y] = world(origin.lon, origin.lat);
            map.centerX = x;
            map.centerY = y;
            map.scale = Math.max(map.scale, 65000);
            draw();
            setText("map-status", `${origin.feature_id}: baseline ${cost(result.baseline_cost_seconds)}, selected ${cost(result.estimated_cost_seconds)}. Graph estimate only.`);
        });
        const changed = result.newly_disconnected ? "Newly disconnected" :
            `${cost(result.estimated_cost_seconds)}${result.cost_delta_seconds && result.cost_delta_seconds > 0.05 ? ` (+${number(result.cost_delta_seconds, 0)} s)` : ""}`;
        name.append(button);
        tr.append(name, textCell(cost(result.baseline_cost_seconds)), textCell(changed, result.newly_disconnected ? "status-disconnected" : ""), textCell(`${origin.designed_stability.reachable_members}/${origin.designed_stability.designed_members} reachable`));
        body.append(tr);
    }
}
function renderHospitals() {
    const target = element("hospital-list");
    target.replaceChildren();
    const inBounds = about.hospitals.filter(row => row.status === "snapped").length;
    setText("hospital-count", `${inBounds} inside study area`);
    setText("hospital-note", about.mode === "synthetic" ?
        "All facilities and positions are invented fixtures." :
        "Facility positions are snapped road nodes. Operating status in 2019 is unknown.");
    for (const hospital of about.hospitals) {
        const item = document.createElement("div");
        item.className = "hospital-item";
        const title = document.createElement("strong");
        title.textContent = hospital.feature_id;
        const state = document.createElement("span");
        state.textContent = hospital.status === "snapped" ?
            `Snapped ${number(hospital.nearest_distance_m, 1)} m · ${hospital.source_date} list` :
            `Outside study area · ${hospital.source_date} list`;
        item.append(title, state);
        target.append(item);
    }
}
function renderSources() {
    const source = element("source-dates");
    source.replaceChildren();
    const fields = [["Road snapshot", about.dates.road_snapshot_utc ?? "—"],
        ["Radar image", about.dates.radar ?? "—"], ["Optical image", about.dates.optical ?? "—"],
        ["Independent trace", about.dates.independent_trace ?? "—"],
        ["Hospital list", about.dates.hospital_snapshot ?? "—"]];
    for (const [name, value] of fields) {
        const dt = document.createElement("dt");
        dt.textContent = name;
        const dd = document.createElement("dd");
        dd.textContent = value;
        source.append(dt, dd);
    }
    const c = about.coverage;
    setText("coverage-copy", `${number(c.segments_intersecting_pilot)}/${number(c.physical_segments_total)} road segments intersect one image tile. ${number(c.outside_pilot_segments)} have no image evidence.`);
    const limits = element("limits-list");
    limits.replaceChildren(...about.interpretation.map(value => {
        const li = document.createElement("li");
        li.textContent = value;
        return li;
    }));
}
function detailGrid(items) {
    const grid = document.createElement("div");
    grid.className = "detail-grid";
    for (const [key, value] of items) {
        const label = document.createElement("span");
        label.textContent = key;
        const entry = document.createElement("span");
        entry.textContent = value;
        grid.append(label, entry);
    }
    return grid;
}
async function inspectRoad(id, open = true) {
    if (!roadIds.has(id)) {
        roadDetail.replaceChildren(document.createTextNode(`Unknown road segment ID: ${id}`));
        return;
    }
    const row = await json(`/api/v1/roads/${encodeURIComponent(id)}`);
    selectedRoad = id;
    const h3 = document.createElement("h3");
    h3.textContent = id;
    const state = document.createElement("p");
    state.className = "detail-state" + (row.evidence ? "" : " unknown");
    state.textContent = row.evidence ? "Inside image footprint · sampled evidence" :
        "Outside image footprint · water status unknown";
    const closed = selected ? closedIds.has(id) : false;
    const fields = [["Road class", row.road_class], ["Length", `${number(row.length_m, 1)} m`],
        ["Bridge / tunnel tags", `${row.bridge ? "Bridge" : "No bridge tag"} / ${row.tunnel ? "Tunnel" : "No tunnel tag"}`],
        ["Selected graph case", closed ? "Assumed removed" : "Retained in graph"]];
    if (row.evidence) {
        fields.push(["Source-supported", `${number(row.evidence.source_supported_m, 1)} m`], ["Unknown length", `${number(row.evidence.source_unknown_m, 1)} m`], ["Quality-retained", `${number(row.evidence.retained_m, 1)} m`], ["Water overlap on support", `${number(row.evidence.source_supported_water_m_by_scenario[selected?.scenario_id ?? ""], 1)} m`], ["Water overlap retained", `${number(row.evidence.retained_water_m_by_scenario[selected?.scenario_id ?? ""], 1)} m`]);
    }
    const warning = document.createElement("p");
    warning.className = "detail-warning";
    warning.textContent = "Overlap and graph removal are scenario evidence only. Water depth, elevation, passability and observed closure are unknown.";
    roadDetail.replaceChildren(h3, state, detailGrid(fields), warning);
    if (open) {
        drawer.hidden = false;
        element("open-drawer").hidden = true;
    }
    draw();
}
function wire() {
    scenarioSelect.addEventListener("change", () => updateAvailable("scenario"));
    policySelect.addEventListener("change", () => updateAvailable("policy"));
    bridgeSelect.addEventListener("change", () => updateAvailable("bridge"));
    for (const id of ["layer-water", "layer-roads", "layer-origins"])
        element(id).addEventListener("change", draw);
    element("fit-study").addEventListener("click", () => fit(about.study_bounds));
    element("fit-tile").addEventListener("click", () => fit(about.tile_bounds));
    element("zoom-in").addEventListener("click", () => zoom(1.4));
    element("zoom-out").addEventListener("click", () => zoom(1 / 1.4));
    for (const [button, field] of [["sort-origin", "origin"], ["sort-baseline", "baseline"],
        ["sort-selected", "selected"]]) {
        element(button).addEventListener("click", () => {
            sortAscending = sortBy === field ? !sortAscending : true;
            sortBy = field;
            renderOrigins();
        });
    }
    element("open-drawer").addEventListener("click", () => { drawer.hidden = false; element("open-drawer").hidden = true; });
    element("close-drawer").addEventListener("click", () => { drawer.hidden = true; element("open-drawer").hidden = false; });
    element("road-search").addEventListener("submit", event => {
        event.preventDefault();
        void inspectRoad(element("road-id").value.trim()).catch(showError);
    });
    for (const [button, key] of [["inspect-exposed", "exposed_link"], ["inspect-bridge", "bridge_overlap"],
        ["inspect-outside", "outside_footprint_link"]]) {
        const control = element(button);
        const id = about.inspection_ids[key];
        if (!id)
            control.hidden = true;
        else
            control.addEventListener("click", () => { void inspectRoad(id).catch(showError); });
    }
    canvas.addEventListener("pointerdown", event => {
        const [x, y] = mapPoint(event);
        pointerStart = { x, y, centerX: map.centerX, centerY: map.centerY, moved: false };
        canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", event => {
        if (!pointerStart)
            return;
        const [x, y] = mapPoint(event);
        if (Math.hypot(x - pointerStart.x, y - pointerStart.y) > 3)
            pointerStart.moved = true;
        if (pointerStart.moved) {
            map.centerX = pointerStart.centerX - (x - pointerStart.x) / map.scale;
            map.centerY = pointerStart.centerY - (y - pointerStart.y) / map.scale;
            draw();
        }
    });
    canvas.addEventListener("pointerup", event => {
        if (!pointerStart)
            return;
        const moved = pointerStart.moved;
        pointerStart = null;
        if (!moved) {
            const [x, y] = mapPoint(event);
            const id = pickRoad(x, y);
            if (id)
                void inspectRoad(id).catch(showError);
        }
    });
    canvas.addEventListener("pointercancel", () => { pointerStart = null; });
    canvas.addEventListener("wheel", event => {
        event.preventDefault();
        const [x, y] = mapPoint(event);
        zoom(event.deltaY < 0 ? 1.25 : 0.8, x, y);
    }, { passive: false });
    new ResizeObserver(resize).observe(canvas);
}
async function start() {
    const [caseInfo, geometry] = await Promise.all([
        json("/api/v1/about"),
        json("/api/v1/roads"),
    ]);
    about = caseInfo;
    roads = geometry.roads;
    roadIds = new Set(roads.map(row => row[0]));
    evidenceIds = new Set(geometry.evidence_segment_ids);
    setText("case-title", about.case_title);
    const badge = element("mode-badge");
    badge.textContent = about.mode === "synthetic" ? "SYNTHETIC · INVENTED" : "LOCAL RESEARCH CASE";
    badge.classList.toggle("synthetic", about.mode === "synthetic");
    const scenarioIds = distinct(about.scenario_options.map(row => row.scenario_id));
    scenarioIds.sort((a, b) => {
        const rank = (value) => value === about.primary_scenario_id ? 0 :
            value === "threshold_0.50" ? 1 : value.startsWith("m3_") ? 2 : 3;
        return rank(a) - rank(b) || a.localeCompare(b);
    });
    options(scenarioSelect, scenarioIds.map(value => [value, labelForScenario(value)]), about.primary_scenario_id);
    renderSources();
    renderHospitals();
    wire();
    resize();
    fit(about.study_bounds);
    updateAvailable();
}
void start().catch(showError);
