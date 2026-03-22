"""
Raster I/O Module
- Load GeoTIFF (classified or raw reflectance)
- Auto-reproject and resample two rasters to a shared grid
- K-means classification for raw multi-band imagery
"""

import numpy as np
import tempfile, os, warnings
from typing import Tuple


class RasterLoader:
    """Handles loading, aligning, and optionally classifying rasters."""

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, path: str, fmt: str = "classified",
             n_classes: int = 7) -> Tuple[np.ndarray, dict]:
        """
        Load a GeoTIFF and return (classified_array, meta).

        fmt:
          'classified' – pixel values are integer class IDs (read band 1)
          'raw'        – multi-band reflectance; apply K-means to classify
        """
        try:
            import rasterio
        except ImportError:
            raise RuntimeError("rasterio is required for raster loading.")

        with rasterio.open(path) as src:
            meta = src.meta.copy()
            if fmt == "classified":
                arr = src.read(1).astype(np.int32)
            else:
                # Multi-band raw: read all bands
                bands = src.read().astype(np.float32)   # (C, H, W)
                arr = self._classify_raw(bands, n_classes)

        return arr, meta

    # ── Align ─────────────────────────────────────────────────────────────────

    def align(self, arr_a: np.ndarray, arr_b: np.ndarray,
              meta_ref: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        If arr_b has a different shape, reproject arr_b to match arr_a's grid
        using nearest-neighbour resampling (preserves integer class IDs).

        When called from the upload endpoint both rasters have already been
        loaded via rasterio and the meta belongs to raster A.  When GEE
        downloads both at the same scale they should already be aligned —
        this is a safety fallback.
        """
        if arr_a.shape == arr_b.shape:
            return arr_a, arr_b, meta_ref

        # If shapes differ we need rasterio to warp
        try:
            import rasterio
            from rasterio.transform import from_bounds
            from rasterio.warp import reproject, Resampling
        except ImportError:
            # Fallback: just resize with numpy
            from scipy.ndimage import zoom
            zy = arr_a.shape[0] / arr_b.shape[0]
            zx = arr_a.shape[1] / arr_b.shape[1]
            arr_b = zoom(arr_b.astype(float), (zy, zx), order=0).astype(np.int32)
            return arr_a, arr_b, meta_ref

        with tempfile.TemporaryDirectory() as tmp:
            ref_path = os.path.join(tmp, "ref.tif")
            src_path = os.path.join(tmp, "src.tif")
            out_path = os.path.join(tmp, "out.tif")

            # Write reference
            m = meta_ref.copy()
            m.update(dtype=rasterio.int32, count=1)
            with rasterio.open(ref_path, "w", **m) as ds:
                ds.write(arr_a[np.newaxis].astype(np.int32))

            # Write source (arr_b) — use same meta but it may have different shape
            m2 = m.copy()
            m2.update(height=arr_b.shape[0], width=arr_b.shape[1])
            with rasterio.open(src_path, "w", **m2) as ds:
                ds.write(arr_b[np.newaxis].astype(np.int32))

            # Warp src to ref
            with rasterio.open(ref_path) as ref_ds:
                with rasterio.open(src_path) as src_ds:
                    out_arr = np.zeros_like(arr_a, dtype=np.int32)
                    reproject(
                        source      = rasterio.band(src_ds, 1),
                        destination = out_arr,
                        src_transform  = src_ds.transform,
                        src_crs        = src_ds.crs,
                        dst_transform  = ref_ds.transform,
                        dst_crs        = ref_ds.crs,
                        resampling     = Resampling.nearest,
                    )

        return arr_a, out_arr, meta_ref

    # ── Private: K-means classification ───────────────────────────────────────

    def _classify_raw(self, bands: np.ndarray, n_classes: int) -> np.ndarray:
        """
        K-means spectral classification of a multi-band raster.
        Input:  bands (C, H, W) float32
        Output: classified (H, W) int32, class IDs 1..n_classes
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            raise RuntimeError("scikit-learn is required for raw band classification.")

        C, H, W = bands.shape
        pixels = bands.reshape(C, -1).T    # (H*W, C)

        # Mask nodata (NaN or zero across all bands)
        valid_mask = ~np.all(pixels == 0, axis=1) & ~np.any(np.isnan(pixels), axis=1)

        km = MiniBatchKMeans(n_clusters=n_classes, random_state=42, batch_size=4096)
        labels = np.zeros(H * W, dtype=np.int32)

        if valid_mask.sum() > 0:
            km.fit(pixels[valid_mask])
            labels[valid_mask] = km.predict(pixels[valid_mask]) + 1  # 1-indexed

        # Sort clusters by mean NDVI (if ≥2 bands) so class 1 ≈ vegetation
        if C >= 2:
            labels = self._sort_by_ndvi(labels, bands, n_classes, valid_mask)

        return labels.reshape(H, W)

    def _sort_by_ndvi(self, labels, bands, n_classes, valid_mask):
        """Re-order cluster IDs so higher IDs = greener (higher NDVI proxy)."""
        C, H, W = bands.shape
        # Use bands[0] as red proxy, bands[1] as NIR proxy if available
        red = bands[0].ravel()
        nir = bands[min(1, C-1)].ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0)

        mean_ndvi = {}
        for cid in range(1, n_classes + 1):
            mask = (labels == cid) & valid_mask
            mean_ndvi[cid] = float(ndvi[mask].mean()) if mask.sum() > 0 else -1

        # Sort ascending: class 1 = least vegetation
        sorted_ids = sorted(mean_ndvi, key=lambda k: mean_ndvi[k])
        remap = {old: new + 1 for new, old in enumerate(sorted_ids)}
        return np.vectorize(remap.get)(labels).astype(np.int32)
