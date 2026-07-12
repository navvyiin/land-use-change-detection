# Land Use Change Detection Dashboard

A full-stack geospatial platform for detecting, quantifying, and visualising land cover change between two time periods, applied to the Western Ghats coffee-forest landscape of Chikkamagaluru District, Karnataka.

## What it does

Ingests MODIS MCD12Q1 (2013) and Dynamic World v1 (2022) GeoTIFF exports from Google Earth Engine at 30m resolution (3,637 × 4,614 px), remaps both to a unified 7-class land cover scheme, and identifies 446,934 ha of land cover transition across the district. The pipeline handles automated CRS reprojection and resampling so that rasters from different sources and native resolutions can be compared directly, and supports both pre-classified inputs and raw multi-band imagery (classified on the fly via K-means clustering).

Ten statistical analyses run on the resulting transition data:

| Analysis | Purpose |
|---|---|
| Transition change matrix | Which classes converted to which |
| Markov chain (steady-state + mixing time) | Long-run equilibrium land cover distribution |
| Cohen's Kappa | Classification agreement beyond chance |
| Moran's I (queen contiguity) | Whether change is spatially clustered or random |
| FRAGSTATS-style landscape metrics | Patch density, edge density, fragmentation |
| Shannon entropy + KL/Jensen-Shannon divergence | Landscape diversity and how it shifted between periods |
| Pontius net vs. swap decomposition | How much "change" is genuine conversion vs. same-class relocation |
| Chi-square test | Whether transitions differ from what random reshuffling would produce |
| FAO compound annual rate of change | Standardised year-on-year change rate |
| Class-level vulnerability index | Low → Critical risk ranking per land cover class |

Google Earth Engine's Python API auto-selects the appropriate dataset by year (Dynamic World for 2015–present, MODIS for 2001–2014, ESA WorldCover for 2021), with Nominatim geocoding for place-name queries. Outputs: a multi-page PDF statistical report, an 8-section CSV data bundle, and a tagged GeoTIFF change map, all downloadable from the dashboard.

## Why ten separate analyses instead of one

Each method answers a different question that a single change-detection metric can't. A change matrix tells you *what* converted to *what*, but not whether that pattern is statistically distinguishable from noise (chi-square), whether it's spatially clustered (Moran's I), or whether it represents net habitat loss versus classification noise relocating pixels between similar classes (Pontius decomposition). Running all ten against the same transition data was the point — it's a deliberate contrast with dashboards that report a single change percentage without interrogating whether that number means anything.

## Tech stack

Python, FastAPI, Rasterio, NumPy, SciPy, Scikit-learn, Google Earth Engine API, Chart.js. Backend deployed on Render, frontend on GitHub Pages.

## Known limitations

- Accuracy is bounded by the source datasets' own classification error — MODIS 500m-class products and Dynamic World disagree at land cover boundaries more often than either disagrees with ground truth, and this isn't independently validated against field data here.
- The K-means fallback for raw imagery is unsupervised and will not reliably separate spectrally similar classes (e.g., agroforestry vs. dense agriculture) without manual review of the resulting clusters.

## Setup

```bash
git clone https://github.com/navvyiin/land-use-change-detection-dashboard.git
cd land-use-change-detection-dashboard
pip install -r backend/requirements.txt
earthengine authenticate
uvicorn backend.main:app --reload
```
