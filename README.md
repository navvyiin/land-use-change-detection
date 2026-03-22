# Land Use Change Detection Dashboard
### Western Ghats · Agroforestry Encroachment Research

A full-stack geospatial system for detecting, quantifying, and visualising land cover change between two time periods. Built for spatial analysis of agroforestry encroachment in the Western Ghats (Chikkamagaluru District), but applicable to any region and dataset.

---

## Project Structure

```
land_use_change/
│
├── backend/
│   ├── main.py               ← FastAPI app (all endpoints)
│   ├── raster_io.py          ← GeoTIFF loading, alignment, K-means classification
│   ├── gee_handler.py        ← Google Earth Engine integration
│   ├── statistics_engine.py  ← All 10 statistical analyses
│   ├── analysis.py           ← Change map computation + matplotlib rendering
│   ├── export_handler.py     ← PDF report, CSV bundle, GeoTIFF export
│   ├── data_generator.py     ← Synthetic demo data (Chikkamagaluru-calibrated)
│   └── requirements.txt
│
├── frontend/
│   └── index.html            ← Single-file dashboard (no build step)
│
├── chikkamagaluru_2013.tif   ← Real GEE export (MODIS, classified)
├── chikkamagaluru_2022.tif   ← Real GEE export (Dynamic World, classified)
├── gee_export_script.js      ← GEE Code Editor script to re-export data
└── README.md
```

---

## Quick Start (Windows PowerShell)

Run each command separately — do NOT paste multiple lines at once in PowerShell.

```powershell
# Step 1 — Navigate to backend
cd C:\path\to\files\backend

# Step 2 — Create virtual environment
python -m venv .venv

# Step 3 — Activate it
.venv\Scripts\Activate.ps1
```

If Step 3 gives a script execution error, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

```powershell
# Step 4 — Install dependencies
pip install -r requirements.txt

# Step 5 — Authenticate GEE (one-time only)
earthengine authenticate

# Step 6 — Start the API server
uvicorn main:app --reload --port 8000
```

Then open a **second PowerShell window**:
```powershell
cd C:\path\to\files\frontend
python -m http.server 3000
```

Open your browser at: **http://localhost:3000**

Both windows must stay open while using the dashboard.

---

## Land Cover Class Table

All datasets remapped to this unified 7-class scheme before analysis:

| ID | Class | Hex Colour | GEE MODIS Source | GEE DW Source |
|----|-------|------------|-----------------|---------------|
| 1 | Forest | `#2d6a4f` | IGBP 1–5 | label=1 (trees) |
| 2 | Shrubland | `#74c69d` | IGBP 6–7 | label=5 (shrub) |
| 3 | Cropland | `#c9a227` | IGBP 12,14 | label=4 (crops) |
| 4 | Grassland | `#95d5b2` | IGBP 8–10 | label=2 (grass) |
| 5 | Built-up | `#bc4749` | IGBP 13 | label=6 (built) |
| 6 | Water/Wetland | `#4895ef` | IGBP 11,17 | label=0,3 (water/flooded) |
| 7 | Bare/Snow | `#d4a373` | IGBP 15–16 | label=7,8 (bare/snow) |

Pre-classified rasters you upload must use these integer IDs in band 1.

---

## Statistical Analyses (10)

| # | Analysis | Method | Key Outputs |
|---|----------|--------|-------------|
| 1 | **Change Matrix** | Cross-tabulation | Transition counts, ha, % per row |
| 2 | **Markov Chain** | Row-stochastic P matrix | Steady-state, mixing time, persistence |
| 3 | **Accuracy / Kappa** | Cohen's κ | OA, κ, per-class PA/UA/F1 |
| 4 | **Moran's I** | Queen contiguity weights | I, Z-score, p-value, autocorrelation |
| 5 | **Landscape Metrics** | FRAGSTATS-style | NP, MPS, LPI, PD, ED, Fragmentation, Contagion |
| 6 | **Information Theory** | Shannon entropy | H(A), H(B), KL divergence, JS divergence, redundancy |
| 7 | **Pontius Decomposition** | Net vs Swap | Systematic vs reciprocal change per class |
| 8 | **Chi-Square** | Independence test | χ², Cramér's V effect size |
| 9 | **Rate of Change** | FAO compound formula | Annual rate %/yr, half-life, doubling time |
| 10 | **Vulnerability Index** | Weighted loss rate | Risk level (Low/Moderate/High/Critical) |

---

## GEE Dataset Auto-Selection

When fetching data directly from GEE, the system auto-selects by year:

| Year Range | Dataset | Native Resolution |
|------------|---------|-----------------|
| 2015–present | Dynamic World v1 | 10 m |
| 2001–2014 | MODIS MCD12Q1 (IGBP) | 500 m |
| 2021 only | ESA WorldCover v200 | 10 m |

All are resampled to **30 m** (Landsat-scale) on download.

---

## API Endpoints

```
GET  /                           Health check
GET  /gee/status                 GEE authentication status
GET  /gee/dataset-info           Preview dataset selection for two years
POST /demo                       Load synthetic demo (Chikkamagaluru-calibrated, 512×512)
POST /upload                     Upload two GeoTIFFs (multipart/form-data)
POST /gee/fetch                  Fetch classified rasters from GEE

GET  /summary/{sid}              Area summary
GET  /area-stats/{sid}           Per-class area stats (ha)
GET  /stats/all/{sid}            All 10 statistics in one call
GET  /stats/change-matrix/{sid}
GET  /stats/markov/{sid}
GET  /stats/accuracy/{sid}
GET  /stats/morans-i/{sid}
GET  /stats/landscape/{sid}
GET  /stats/information/{sid}
GET  /stats/pontius/{sid}
GET  /stats/chi-square/{sid}
GET  /stats/rate-of-change/{sid}
GET  /stats/vulnerability/{sid}

GET  /map/{sid}/{type}           PNG image (raster_2010 | raster_2020 | change | gain_loss)
GET  /export/pdf/{sid}           Multi-page PDF report
GET  /export/csv/{sid}           8-section CSV data bundle
GET  /export/geotiff/{sid}       Change map GeoTIFF
```

Upload parameters (`POST /upload`):
- `raster_a`, `raster_b` — GeoTIFF files
- `year_a`, `year_b` — integers
- `format_a`, `format_b` — `"classified"` or `"raw"`
- `n_classes` — integer (for K-means, only used when format is `"raw"`)

---

## Raster Input Formats

### Pre-classified
Band 1 contains integer class IDs 1–7. Values of 0 are treated as NoData.

### Raw reflectance
Multi-band GeoTIFF (e.g. Landsat bands). The system applies MiniBatchKMeans clustering, then sorts clusters by NDVI proxy so class 1 = least vegetated. You choose the number of classes (2–15).

### Auto-alignment
If the two rasters have different CRS, resolution, or extent, the system automatically reprojects and resamples raster B to match raster A using nearest-neighbour (preserves class IDs).

---

## Re-exporting from GEE

To re-run the GEE export (e.g. for different years or area):

1. Open `gee_export_script.js`
2. Paste into [code.earthengine.google.com](https://code.earthengine.google.com)
3. Modify years or district name if needed
4. Click **Run**, then go to the **Tasks** tab and click **RUN** on each task
5. Files export to `My Drive → GEE_LandUse/`

---

## Requirements

```
fastapi==0.115.6
uvicorn[standard]==0.32.1
python-multipart==0.0.19
numpy==1.26.4
scipy==1.13.1
matplotlib==3.9.2
rasterio==1.3.11
scikit-learn==1.5.2
reportlab==4.2.5
geopy==2.4.1
earthengine-api==0.1.414
requests==2.32.3
```

**System dependencies:**
- Python ≥ 3.10
- rasterio requires GDAL — on Ubuntu/Debian: `apt install gdal-bin libgdal-dev`
- On Windows, rasterio installs with bundled GDAL (no extra step needed)
- GEE features require a Google Earth Engine account

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `&&` not valid in PowerShell | Run commands one at a time |
| `.venv\Scripts\Activate.ps1` blocked | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` first |
| `Could not import module "main"` | Make sure you `cd backend` before running uvicorn |
| `cd ..\frontend` fails | Use full path: `cd C:\...\files\frontend` |
| GEE not authenticated | Run `earthengine authenticate` once; credentials persist |
| Analysis is slow | Normal for full-size rasters (3637×4614). Demo data (512×512) is instant |
| CORS error in browser | Ensure backend is running on port 8000 and frontend on port 3000 |
