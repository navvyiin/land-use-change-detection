"""
Land Use Change Detection — FastAPI Backend
Endpoints: upload rasters, GEE fetch, full statistical analysis, export
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn, io, os, tempfile, uuid, json
from typing import Optional

from raster_io import RasterLoader
from analysis import ChangeAnalyzer
from statistics_engine import LandUseStatistics
from gee_handler import GEEHandler, UNIFIED_CLASSES
from export_handler import ExportHandler
from data_generator import generate_sample_rasters

app = FastAPI(title="Land Use Change API", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ── Session Store ─────────────────────────────────────────────────────────────
sessions: dict[str, dict] = {}   # session_id → {analyzer, stats, meta, ...}

# ── GEE handler (singleton) ───────────────────────────────────────────────────
gee = GEEHandler()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def _get(sid: str) -> dict:
    if sid not in sessions:
        raise HTTPException(404, f"Session '{sid}' not found.")
    return sessions[sid]

def _build_session(arr_a, arr_b, meta, year_a: int, year_b: int,
                   classes: dict = None, sid: str = None) -> str:
    sid = sid or str(uuid.uuid4())[:8]
    classes = classes or UNIFIED_CLASSES
    n_years = max(year_b - year_a, 1)

    loader    = RasterLoader()
    arr_a, arr_b, meta = loader.align(arr_a, arr_b, meta)

    analyzer  = ChangeAnalyzer(arr_a, arr_b, meta, classes)
    analyzer.run_all()

    stats     = LandUseStatistics(arr_a, arr_b, classes, n_years=n_years)
    stats.run_all()

    sessions[sid] = {
        "analyzer":  analyzer,
        "stats":     stats,
        "meta":      meta,
        "classes":   classes,
        "year_a":    year_a,
        "year_b":    year_b,
        "n_years":   n_years,
    }
    return sid


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS / HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0"}

@app.get("/gee/status")
def gee_status():
    ok, msg = gee.check_auth()
    return {"authenticated": ok, "message": msg}


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/demo")
def load_demo():
    a, b, meta = generate_sample_rasters(size=512)
    sid = _build_session(a, b, meta, year_a=2010, year_b=2020)
    return {"session_id": sid, "message": "Demo 512×512 synthetic landscape loaded"}


# ═══════════════════════════════════════════════════════════════════════════════
# RASTER UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_rasters(
    raster_a:     UploadFile = File(...),
    raster_b:     UploadFile = File(...),
    year_a:       int        = Form(2010),
    year_b:       int        = Form(2020),
    format_a:     str        = Form("classified"),   # 'classified' | 'raw'
    format_b:     str        = Form("classified"),
    n_classes:    int        = Form(7),
):
    """
    Accept two GeoTIFF rasters (pre-classified or raw reflectance bands).
    Auto-reprojects and resamples if CRS/resolution differs.
    """
    import rasterio, numpy as np
    loader = RasterLoader()

    with tempfile.TemporaryDirectory() as tmp:
        path_a = os.path.join(tmp, "a.tif")
        path_b = os.path.join(tmp, "b.tif")
        open(path_a, "wb").write(await raster_a.read())
        open(path_b, "wb").write(await raster_b.read())

        try:
            arr_a, meta_a = loader.load(path_a, fmt=format_a, n_classes=n_classes)
            arr_b, meta_b = loader.load(path_b, fmt=format_b, n_classes=n_classes)
        except Exception as e:
            raise HTTPException(400, f"Raster load error: {e}")

    sid = _build_session(arr_a, arr_b, meta_a, year_a, year_b)
    return {
        "session_id": sid,
        "shape":      list(arr_a.shape),
        "crs":        str(meta_a.get("crs", "unknown")),
        "year_a":     year_a,
        "year_b":     year_b,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GEE FETCH
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/gee/dataset-info")
def gee_dataset_info(year_a: int, year_b: int):
    """Preview which datasets will be used for each year."""
    try:
        ds_a = gee.select_dataset(year_a)
        ds_b = gee.select_dataset(year_b)
        return {
            "year_a": {"year": year_a, "dataset": ds_a["name"],
                       "collection": ds_a["collection"],
                       "native_resolution": ds_a["resolution_native"],
                       "description": ds_a["description"]},
            "year_b": {"year": year_b, "dataset": ds_b["name"],
                       "collection": ds_b["collection"],
                       "native_resolution": ds_b["resolution_native"],
                       "description": ds_b["description"]},
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/gee/fetch")
async def gee_fetch(
    year_a:    int  = Form(...),
    year_b:    int  = Form(...),
    place:     Optional[str]   = Form(None),
    bbox_minx: Optional[float] = Form(None),
    bbox_miny: Optional[float] = Form(None),
    bbox_maxx: Optional[float] = Form(None),
    bbox_maxy: Optional[float] = Form(None),
):
    """
    Fetch land cover rasters from GEE for two years.
    Area may be specified as a place name or bounding box.
    Downloads at 30m and runs full analysis pipeline.
    """
    ok, msg = gee.check_auth()
    if not ok:
        raise HTTPException(403, f"GEE not authenticated: {msg}")

    # Resolve area
    try:
        if place:
            bbox = gee.geocode(place)
        elif all(v is not None for v in [bbox_minx, bbox_miny, bbox_maxx, bbox_maxy]):
            bbox = [bbox_minx, bbox_miny, bbox_maxx, bbox_maxy]
        else:
            raise HTTPException(400, "Provide either 'place' or all four bbox fields.")

        arr_a, arr_b, meta = gee.fetch_classified(
            bbox=bbox, year_a=year_a, year_b=year_b, scale=30
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"GEE fetch error: {e}")

    sid = _build_session(arr_a, arr_b, meta, year_a, year_b, classes=UNIFIED_CLASSES)
    return {
        "session_id": sid,
        "bbox":       bbox,
        "shape":      list(arr_a.shape),
        "year_a":     year_a,
        "year_b":     year_b,
        "dataset_a":  gee.select_dataset(year_a)["name"],
        "dataset_b":  gee.select_dataset(year_b)["name"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/summary/{sid}")
def summary(sid: str):
    s = _get(sid)
    return s["analyzer"].summary()

@app.get("/area-stats/{sid}")
def area_stats(sid: str):
    return _get(sid)["analyzer"].area_stats()

@app.get("/change-map-stats/{sid}")
def change_map_stats(sid: str):
    return _get(sid)["analyzer"].change_map_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/stats/change-matrix/{sid}")
def stats_change_matrix(sid: str):
    return _get(sid)["stats"].change_matrix_result()

@app.get("/stats/markov/{sid}")
def stats_markov(sid: str):
    return _get(sid)["stats"].markov_result()

@app.get("/stats/accuracy/{sid}")
def stats_accuracy(sid: str):
    return _get(sid)["stats"].accuracy_result()

@app.get("/stats/morans-i/{sid}")
def stats_morans(sid: str):
    return _get(sid)["stats"].morans_i_result()

@app.get("/stats/landscape/{sid}")
def stats_landscape(sid: str):
    return _get(sid)["stats"].landscape_result()

@app.get("/stats/information/{sid}")
def stats_information(sid: str):
    return _get(sid)["stats"].information_result()

@app.get("/stats/pontius/{sid}")
def stats_pontius(sid: str):
    return _get(sid)["stats"].pontius_result()

@app.get("/stats/chi-square/{sid}")
def stats_chi_square(sid: str):
    return _get(sid)["stats"].chi_square_result()

@app.get("/stats/rate-of-change/{sid}")
def stats_roc(sid: str):
    return _get(sid)["stats"].rate_of_change_result()

@app.get("/stats/vulnerability/{sid}")
def stats_vuln(sid: str):
    return _get(sid)["stats"].vulnerability_result()

@app.get("/stats/all/{sid}")
def stats_all(sid: str):
    """Return all statistics in a single call."""
    st = _get(sid)["stats"]
    return {
        "change_matrix":  st.change_matrix_result(),
        "markov":         st.markov_result(),
        "accuracy":       st.accuracy_result(),
        "morans_i":       st.morans_i_result(),
        "landscape":      st.landscape_result(),
        "information":    st.information_result(),
        "pontius":        st.pontius_result(),
        "chi_square":     st.chi_square_result(),
        "rate_of_change": st.rate_of_change_result(),
        "vulnerability":  st.vulnerability_result(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAP IMAGES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/map/{sid}/{map_type}")
def get_map(sid: str, map_type: str):
    az = _get(sid)["analyzer"]
    img_bytes = az.render_map(map_type)
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/export/csv/{sid}")
def export_csv(sid: str):
    s = _get(sid)
    exporter = ExportHandler(s["analyzer"], s["stats"], s["year_a"], s["year_b"])
    data = exporter.to_csv()
    return StreamingResponse(
        io.StringIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=landuse_change_{sid}.csv"}
    )

@app.get("/export/pdf/{sid}")
def export_pdf(sid: str):
    s = _get(sid)
    exporter = ExportHandler(s["analyzer"], s["stats"], s["year_a"], s["year_b"])
    pdf_bytes = exporter.to_pdf()
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=landuse_report_{sid}.pdf"}
    )

@app.get("/export/geotiff/{sid}")
def export_geotiff(sid: str):
    s = _get(sid)
    exporter = ExportHandler(s["analyzer"], s["stats"], s["year_a"], s["year_b"])
    tif_bytes = exporter.to_geotiff()
    return StreamingResponse(
        io.BytesIO(tif_bytes),
        media_type="image/tiff",
        headers={"Content-Disposition": f"attachment; filename=change_map_{sid}.tif"}
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
