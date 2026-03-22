"""
Synthetic raster generator — calibrated to real Chikkamagaluru District data.

Actual distributions (valid pixels only, from GEE exports):

  2013 (MODIS):          2022 (Dynamic World):
  Forest      25.5%      Forest      71.2%   ← dominant story
  Shrubland   28.7%      Shrubland   11.2%
  Cropland    33.9%      Cropland    10.9%
  Grassland   10.6%      Grassland    1.3%
  Built-up     0.2%      Built-up     2.8%
  Water        1.0%      Water        2.6%
  Bare         ~0%       Bare         0.1%

Key transitions observed in real data:
  Shrubland  → Forest    184,265 ha  (coffee estates classified as trees by DW)
  Cropland   → Forest     99,969 ha
  Grassland  → Forest     47,014 ha
  Cropland   → Shrubland  54,304 ha
  Cropland   → Built-up   14,274 ha
"""

import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter


def generate_sample_rasters(size: int = 512) -> Tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(2013)

    def noise(sigma, seed):
        return gaussian_filter(rng.random((size, size)), sigma=sigma * size)

    n1 = noise(0.10, 1)
    n2 = noise(0.06, 2)
    n3 = noise(0.14, 3)
    n4 = noise(0.04, 4)

    # District boundary mask (~52% nodata — irregular ellipse with rough edges)
    Y, X = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    ellipse = ((X - cx) / (size * 0.48)) ** 2 + ((Y - cy) / (size * 0.38)) ** 2
    roughness = gaussian_filter(rng.random((size, size)), sigma=0.04 * size)
    inside = (ellipse < 1.0) & (roughness > 0.18)
    inside = inside & ~(noise(0.05, 99) < 0.06)

    # Year A — 2013 (MODIS proportions)
    a = np.zeros((size, size), dtype=np.int32)
    a[inside] = 3                                          # Cropland base 33.9%
    a[inside & (n1 > 0.41)]           = 2                 # Shrubland 28.7%
    a[inside & (n2 > 0.56)]           = 1                 # Forest 25.5%
    a[inside & (n1 < 0.28) & (n3 < 0.52)] = 4            # Grassland 10.6%
    a[inside & (n3 < 0.04)]           = 6                 # Water 1.0%
    a[inside & (n4 > 0.90) & (n2 < 0.35)] = 5            # Built-up 0.2%

    # Year B — 2022 (Dynamic World proportions)
    b = a.copy()
    c1 = noise(0.07, 11)
    c2 = noise(0.05, 22)

    b[(a == 2) & (c1 > 0.36)]  = 1                       # Shrubland → Forest
    b[(a == 3) & (c2 > 0.42)]  = 1                       # Cropland  → Forest
    b[(a == 3) & (b == 3) & (c1 < 0.30)] = 2             # Cropland  → Shrubland
    b[(a == 4) & (c1 > 0.18)]  = 1                       # Grassland → Forest
    b[(a == 4) & (b == 4)]     = 2                       # Grassland → Shrubland

    # Cropland → Built-up near town centre
    dist_centre = np.sqrt((X - cx)**2 + (Y - cy)**2)
    b[(a == 3) & (b == 3) & (dist_centre < size * 0.18) & (c2 > 0.55)] = 5

    # New water bodies
    b[(a != 6) & (b != 5) & (b != 1) & (n3 < 0.03)] = 6

    # Small bare patches (exposed rock ridges)
    b[(a == 4) & (b == 4) & (c2 > 0.80)] = 7

    a[~inside] = 0
    b[~inside] = 0

    meta = {
        "driver": "GTiff", "dtype": "int32", "width": size, "height": size,
        "count": 1, "crs": "EPSG:32643", "nodata": 0,
        "transform": [30.0, 0.0, 763000.0, 0.0, -30.0, 1380000.0],
    }
    return a, b, meta


if __name__ == "__main__":
    a, b, _ = generate_sample_rasters(512)
    CLASSES = {1:"Forest",2:"Shrubland",3:"Cropland",4:"Grassland",5:"Built-up",6:"Water",7:"Bare"}
    for label, arr in [("2013", a), ("2022", b)]:
        valid = arr[arr > 0]; total = valid.size
        print(f"\n{label}:")
        for cid in range(1, 8):
            n = (valid == cid).sum()
            print(f"  {CLASSES[cid]:<16} {n/total*100:5.1f}%")