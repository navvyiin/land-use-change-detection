# TerraShift
### An Open-Source Earth Observation Platform for Multi-Temporal Land Cover Change Detection, Spatial Statistics, and Environmental Intelligence

> A full-stack geospatial analytics platform that automates land-cover change detection, statistical landscape analysis, and environmental reporting using multi-source Earth observation data.

TerraShift is an end-to-end computational framework designed to analyse landscape dynamics across time using satellite imagery.

The platform integrates Google Earth Engine, raster processing, statistical analysis, landscape ecology, Markov modelling, information theory, spatial autocorrelation, automated reporting, and an interactive web dashboard into a single reproducible workflow.

Although demonstrated using agroforestry encroachment in the Western Ghats of India, TerraShift is designed as a general-purpose Earth observation platform capable of analysing land-use change anywhere in the world.

---

# Motivation

Land-use change is one of the dominant drivers of biodiversity loss, climate change, habitat fragmentation, and environmental degradation.

Despite the growing availability of satellite imagery, analysing land-cover dynamics often requires a fragmented workflow spanning multiple desktop GIS applications, remote sensing software, statistical packages, and manual reporting.

TerraShift was developed to unify these disconnected processes into a single computational platform capable of transforming raw Earth observation data into scientifically meaningful environmental intelligence.

Rather than simply generating change maps, TerraShift provides quantitative statistical analyses, automated reports, and decision-support products that enable researchers, planners, and environmental practitioners to investigate landscape transformation at multiple spatial and temporal scales.

---

# Research Objectives

TerraShift is designed to

- detect multi-temporal land-cover transitions
- quantify landscape change
- evaluate fragmentation patterns
- model long-term landscape dynamics
- measure spatial autocorrelation
- assess environmental vulnerability
- generate publication-ready reports
- support evidence-based environmental planning

---

# Key Features

## Multi-Temporal Earth Observation

- Google Earth Engine integration
- MODIS
- Dynamic World
- ESA WorldCover
- Multi-year raster comparison
- Automated data harmonisation

---

## Intelligent Raster Processing

- Automatic reprojection
- Resolution harmonisation
- Spatial alignment
- CRS standardisation
- GeoTIFF validation
- NoData handling

---

## Land Cover Classification

Supports both

### Pre-classified rasters

and

### Raw multispectral imagery

using automated MiniBatch K-Means clustering.

---

## Statistical Landscape Analysis

TerraShift performs ten complementary analyses including

- Change Matrix
- Markov Chain Modelling
- Cohen's Kappa
- Moran's I
- Landscape Metrics
- Information Theory
- Pontius Decomposition
- Chi-Square Analysis
- Annual Rate of Change
- Environmental Vulnerability Index

---

## Automated Reporting

Generate

- Publication-quality figures
- Multi-page PDF reports
- CSV summaries
- GeoTIFF outputs

without additional processing.

---

## Interactive Dashboard

A browser-based interface enables users to

- upload raster datasets
- fetch imagery directly from Google Earth Engine
- visualise change maps
- inspect statistical outputs
- download analytical reports

without requiring desktop GIS software.

---

# Computational Workflow

```text
           Earth Observation Data
                    │
                    ▼
        Google Earth Engine / Upload
                    │
                    ▼
      Raster Harmonisation & Alignment
                    │
                    ▼
      Land Cover Classification
                    │
                    ▼
      Multi-Temporal Change Detection
                    │
                    ▼
────────────────────────────────────────────
│      Statistical Analysis Engine         │
│                                          │
│ Change Matrix                            │
│ Markov Chains                            │
│ Moran's I                                │
│ Landscape Metrics                        │
│ Information Theory                       │
│ Vulnerability Index                      │
│ Pontius Decomposition                    │
────────────────────────────────────────────
                    │
                    ▼
      Interactive Dashboard
                    │
                    ▼
 Automated Reports & Spatial Products
```

---

# System Architecture

```text
Frontend (HTML Dashboard)
          │
          ▼
FastAPI Backend
          │
──────────────────────────────────────
│ Raster Processing                  │
│ Earth Engine Integration           │
│ Statistical Engine                 │
│ Reporting Engine                   │
│ Export Services                    │
──────────────────────────────────────
          │
          ▼
 GeoTIFF • PDF • CSV • Maps
```

---

# Repository Structure

```text
TerraShift/

├── backend/
│   ├── main.py
│   ├── raster_io.py
│   ├── gee_handler.py
│   ├── statistics_engine.py
│   ├── analysis.py
│   ├── export_handler.py
│   ├── data_generator.py
│   └── requirements.txt
│
├── frontend/
│   └── index.html
│
├── docs/
│   ├── methodology.md
│   ├── architecture.md
│   ├── validation.md
│   ├── benchmark.md
│   ├── statistics.md
│   └── references.md
│
├── assets/
│   ├── screenshots/
│   ├── workflow.gif
│   └── architecture.png
│
├── gee_export_script.js
└── README.md
```

---

# Methodology

TerraShift follows a fully reproducible analytical pipeline consisting of six stages.

## 1. Data Acquisition

Users may

- upload GeoTIFF rasters

or

- retrieve classified imagery directly from Google Earth Engine.

The platform automatically selects the most appropriate dataset according to acquisition year.

---

## 2. Raster Harmonisation

Input datasets are standardised through

- coordinate transformation
- reprojection
- spatial alignment
- nearest-neighbour resampling
- NoData management

ensuring that both temporal snapshots share identical spatial geometry before analysis.

---

## 3. Land Cover Processing

Depending on input format,

the platform either

- accepts pre-classified land-cover rasters

or

- performs unsupervised MiniBatch K-Means classification on raw multispectral imagery.

Class labels are normalised into a unified seven-class land-cover ontology.

---

## 4. Multi-Temporal Change Detection

Pixel-wise comparisons generate

- transition matrices
- gain-loss maps
- persistence maps
- class conversion statistics

forming the basis for subsequent analyses.

---

## 5. Statistical Landscape Analysis

Ten analytical modules quantify

- temporal persistence
- spatial dependence
- fragmentation
- uncertainty
- landscape entropy
- ecological vulnerability

using established methods from spatial statistics, landscape ecology, and information theory.

---

## 6. Reporting

Results are automatically exported as

- PDF reports
- CSV summaries
- GeoTIFF rasters
- publication-ready figures

without requiring external GIS software.

---

# Statistical Framework

TerraShift implements ten analytical modules.

| Module | Purpose |
|---------|---------|
| Change Matrix | Quantifies land-cover transitions |
| Markov Chain | Models long-term transition dynamics |
| Accuracy Assessment | Classification agreement |
| Moran's I | Spatial autocorrelation |
| Landscape Metrics | Fragmentation analysis |
| Information Theory | Landscape complexity |
| Pontius Decomposition | Net vs swap change |
| Chi-Square Test | Statistical independence |
| Annual Rate of Change | Temporal dynamics |
| Vulnerability Index | Environmental risk assessment |

---

# Installation

## Requirements

- Python 3.10+
- FastAPI
- GDAL
- Google Earth Engine
- Rasterio

```bash
git clone https://github.com/navvyiin/TerraShift.git

cd TerraShift

python -m venv env

source env/bin/activate

pip install -r backend/requirements.txt
```

Authenticate Google Earth Engine

```bash
earthengine authenticate
```

---

# Running TerraShift

Start the backend

```bash
uvicorn backend.main:app --reload
```

Launch the frontend

```bash
cd frontend

python -m http.server 3000
```

Navigate to

```
http://localhost:3000
```

---

# API Overview

TerraShift exposes REST endpoints for

- raster upload
- Earth Engine integration
- statistical analysis
- report generation
- map rendering
- data export

making the platform suitable for integration into larger geospatial workflows.

---

# Outputs

The platform generates

- Land-cover maps
- Change maps
- Transition matrices
- Markov probabilities
- Spatial autocorrelation statistics
- Landscape metrics
- Information-theoretic measures
- Vulnerability assessments
- PDF reports
- CSV datasets
- GeoTIFF exports

---

# Engineering Challenges

The primary engineering challenge involved harmonising heterogeneous raster datasets originating from different Earth observation products.

Satellite datasets differ substantially in

- spatial resolution
- coordinate reference systems
- classification ontologies
- temporal coverage
- pixel alignment

Building a robust preprocessing engine capable of automatically standardising these differences while preserving analytical validity required significantly more engineering effort than the statistical analyses themselves.

A second challenge involved designing a modular statistical engine capable of integrating methods from spatial statistics, landscape ecology, Markov modelling, and information theory within a consistent computational framework.

---

# Applications

TerraShift can support

- Land Use & Land Cover Change Detection
- Forest Monitoring
- Environmental Impact Assessment
- Biodiversity Conservation
- Climate Change Research
- Landscape Ecology
- Urban Expansion Monitoring
- Agricultural Monitoring
- Geospatial Artificial Intelligence
- Environmental Decision Support

---

# Current Limitations

Current limitations include

- batch-based processing
- single-node execution
- optical imagery dependence
- absence of SAR integration
- no distributed raster computation
- no deep learning segmentation

These represent future development opportunities rather than methodological constraints.

---

# Future Directions

Future releases will explore

- Foundation Models for Earth Observation
- Vision Transformers for semantic segmentation
- Cloud-Optimized GeoTIFF support
- STAC Catalog integration
- GeoParquet compatibility
- Distributed raster processing with Dask
- Kubernetes deployment
- MLflow experiment tracking
- Real-time environmental monitoring
- Multi-temporal forecasting using spatio-temporal transformers

---

# Scientific Significance

TerraShift demonstrates how Earth observation, geospatial data engineering, spatial statistics, landscape ecology, and modern software engineering can be integrated into a unified computational framework for environmental intelligence.

Rather than functioning as a conventional change-detection dashboard, TerraShift formalises land-cover analysis into a reproducible scientific workflow capable of supporting research, environmental monitoring, and evidence-based policy.

---

# Citation

If you use TerraShift in academic work, please cite

```text
Naval Kishore

TerraShift: An Open-Source Earth Observation Platform for Multi-Temporal Land Cover Change Detection, Spatial Statistics, and Environmental Intelligence.

GitHub Repository, 2026.
```

---

# License

Released under the MIT License.

---

# Acknowledgements

TerraShift builds upon the open scientific ecosystem provided by

- Google Earth Engine
- ESA Copernicus Programme
- Dynamic World
- MODIS Land Cover
- GeoPandas
- Rasterio
- GDAL
- NumPy
- SciPy
- scikit-learn
- FastAPI
- Matplotlib

whose contributions continue to advance open Earth observation and geospatial data science.
