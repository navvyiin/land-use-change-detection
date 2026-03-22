"""
Google Earth Engine Handler
- Authentication check
- Dataset auto-selection by year
- Geocoding of place names
- Download classified rasters at 30m to numpy arrays
"""

import numpy as np
import tempfile, os
from typing import Tuple, Optional

# ── Unified land cover classes ────────────────────────────────────────────────
UNIFIED_CLASSES = {
    1: {"name": "Forest",         "color": "#2d6a4f"},
    2: {"name": "Shrubland",      "color": "#74c69d"},
    3: {"name": "Cropland",       "color": "#c9a227"},
    4: {"name": "Grassland",      "color": "#95d5b2"},
    5: {"name": "Built-up",       "color": "#bc4749"},
    6: {"name": "Water/Wetland",  "color": "#4895ef"},
    7: {"name": "Bare/Snow",      "color": "#d4a373"},
}

# ── Dataset registry ──────────────────────────────────────────────────────────
# Priority order: Dynamic World preferred for >=2015; MODIS for 2001-2014
DATASETS = {
    "dynamic_world": {
        "name":             "Dynamic World v1",
        "collection":       "GOOGLE/DYNAMICWORLD/V1",
        "band":             "label",
        "resolution_native": 10,
        "year_range":       (2015, 9999),
        "description":      "Google/Sentinel-2 based near-real-time classification (10 m)",
        # DW: 0=water,1=trees,2=grass,3=flooded_veg,4=crops,5=shrub,6=built,7=bare,8=snow
        "class_remap": {0:6, 1:1, 2:4, 3:6, 4:3, 5:2, 6:5, 7:7, 8:7},
    },
    "modis_mcd12q1": {
        "name":             "MODIS MCD12Q1 (IGBP)",
        "collection":       "MODIS/061/MCD12Q1",
        "band":             "LC_Type1",
        "resolution_native": 500,
        "year_range":       (2001, 2023),
        "description":      "MODIS annual land cover (500 m, IGBP scheme)",
        # IGBP 1-5=forest,6-7=shrub,8-9=savanna/shrub,10=grass,11=wetland,
        # 12=crop,13=urban,14=crop/mosaic,15-16=bare,17=water
        "class_remap": {
            1:1, 2:1, 3:1, 4:1, 5:1,
            6:2, 7:2,
            8:2, 9:4,
            10:4,
            11:6,
            12:3,
            13:5,
            14:3,
            15:7, 16:7,
            17:6,
        },
    },
    "esa_worldcover": {
        "name":             "ESA WorldCover 10m",
        "collection":       "ESA/WorldCover/v200",
        "band":             "Map",
        "resolution_native": 10,
        "year_range":       (2021, 2021),
        "description":      "ESA WorldCover (10 m, available for 2021)",
        # ESA: 10=tree,20=shrub,30=grass,40=crop,50=built,60=bare,70=snow,80=water,90=wetland,95=mangrove,100=moss
        "class_remap": {
            10:1, 20:2, 30:4, 40:3, 50:5, 60:7, 70:7, 80:6, 90:6, 95:1, 100:4,
        },
    },
}


def select_dataset_for_year(year: int) -> dict:
    """Return the most appropriate dataset dict for the given year."""
    if year == 2021:
        return DATASETS["esa_worldcover"]
    if year >= 2015:
        return DATASETS["dynamic_world"]
    if 2001 <= year <= 2023:
        return DATASETS["modis_mcd12q1"]
    raise ValueError(
        f"Year {year} is outside supported range (2001–present). "
        "No suitable GEE land cover dataset available."
    )


class GEEHandler:
    """Wraps all Google Earth Engine interactions."""

    def __init__(self):
        self._ee = None

    # ── Auth ──────────────────────────────────────────────────────────────────

    def check_auth(self) -> Tuple[bool, str]:
        """Return (is_authenticated, message)."""
        try:
            import ee
            ee.Initialize(opt_url="https://earthengine.googleapis.com")
            self._ee = ee
            return True, "Google Earth Engine authenticated successfully."
        except Exception as e:
            return False, (
                f"GEE not authenticated ({e}). "
                "Run `earthengine authenticate` in your terminal, then restart the server."
            )

    def _require_ee(self):
        if self._ee is None:
            ok, msg = self.check_auth()
            if not ok:
                raise RuntimeError(msg)
        return self._ee

    def select_dataset(self, year: int) -> dict:
        return select_dataset_for_year(year)

    # ── Geocoding ─────────────────────────────────────────────────────────────

    def geocode(self, place: str) -> list:
        """
        Geocode a place name to [minx, miny, maxx, maxy] bounding box.
        Uses Nominatim (OSM); no API key required.
        """
        try:
            from geopy.geocoders import Nominatim
            from geopy.exc import GeocoderTimedOut
        except ImportError:
            raise RuntimeError("geopy is required for place geocoding: pip install geopy")

        geo = Nominatim(user_agent="landuse_change_app/2.0")
        try:
            result = geo.geocode(place, exactly_one=True, timeout=10)
        except GeocoderTimedOut:
            raise ValueError(f"Geocoding timed out for '{place}'.")

        if result is None:
            raise ValueError(
                f"Could not geocode '{place}'. "
                "Try a more specific name or provide a bounding box instead."
            )

        bb = result.raw.get("boundingbox")
        if bb:
            miny, maxy, minx, maxx = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        else:
            lat, lon = result.latitude, result.longitude
            delta = 0.25   # ~28 km buffer
            miny, maxy = lat - delta, lat + delta
            minx, maxx = lon - delta, lon + delta

        # Sanity check: cap area to ~200 km² to stay within GEE download limits
        lat_span = maxy - miny
        lon_span = maxx - minx
        if lat_span > 2.0 or lon_span > 2.0:
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            half = 1.0
            minx, maxx = cx - half, cx + half
            miny, maxy = cy - half, cy + half

        return [minx, miny, maxx, maxy]

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def fetch_classified(
        self,
        bbox:  list,
        year_a: int,
        year_b: int,
        scale:  int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Download two classified land cover rasters from GEE.
        Returns (arr_a, arr_b, meta) where arrays are H×W int32.
        """
        ee = self._require_ee()

        minx, miny, maxx, maxy = bbox
        region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

        ds_a = select_dataset_for_year(year_a)
        ds_b = select_dataset_for_year(year_b)

        img_a = self._get_image(ee, ds_a, year_a, region)
        img_b = self._get_image(ee, ds_b, year_b, region)

        # Remap to unified classes
        img_a = self._remap(ee, img_a, ds_a["class_remap"])
        img_b = self._remap(ee, img_b, ds_b["class_remap"])

        arr_a, meta = self._download(ee, img_a, region, scale, "year_a")
        arr_b, _    = self._download(ee, img_b, region, scale, "year_b")

        return arr_a, arr_b, meta

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_image(self, ee, ds: dict, year: int, region) -> object:
        """Build a single-band classified ee.Image for the year."""
        coll   = ds["collection"]
        band   = ds["band"]
        start  = f"{year}-01-01"
        end    = f"{year}-12-31"
        key    = [k for k, v in DATASETS.items() if v == ds][0]

        if key == "dynamic_world":
            return (
                ee.ImageCollection(coll)
                .filterDate(start, end)
                .filterBounds(region)
                .select(band)
                .mode()              # Most common class per pixel
                .clip(region)
                .rename("class")
            )
        elif key == "esa_worldcover":
            return (
                ee.ImageCollection(coll)
                .first()
                .select(band)
                .clip(region)
                .rename("class")
            )
        else:  # MODIS
            return (
                ee.ImageCollection(coll)
                .filterDate(start, end)
                .first()
                .select(band)
                .clip(region)
                .rename("class")
            )

    def _remap(self, ee, image, class_remap: dict):
        """Apply class ID remapping to a single-band ee.Image."""
        from_list = list(class_remap.keys())
        to_list   = [class_remap[k] for k in from_list]
        return image.remap(from_list, to_list, defaultValue=0)

    def _download(self, ee, image, region, scale: int, tag: str
                  ) -> Tuple[np.ndarray, dict]:
        """Download an ee.Image as GeoTIFF to a temp file, return (array, meta)."""
        import requests, rasterio

        url = image.getDownloadURL({
            "scale":  scale,
            "region": region,
            "format": "GEO_TIFF",
            "filePerBand": False,
        })

        resp = requests.get(url, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"GEE download failed ({resp.status_code}): {resp.text[:200]}")

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            f.write(resp.content)
            tmp_path = f.name

        try:
            with rasterio.open(tmp_path) as src:
                arr  = src.read(1).astype(np.int32)
                meta = src.meta.copy()
        finally:
            os.unlink(tmp_path)

        return arr, meta
