// ============================================================
// Land Use Change Detection — Chikkamagaluru District
// Google Earth Engine Export Script
// Run this in https://code.earthengine.google.com
// Exports two classified GeoTIFFs to your Google Drive
// ============================================================

// ── 1. Study Area — Chikkamagaluru District boundary ────────
var district = ee.FeatureCollection("FAO/GAUL/2015/level2")
  .filter(ee.Filter.and(
    ee.Filter.eq("ADM1_NAME", "Karnataka"),
    ee.Filter.eq("ADM2_NAME", "Chikmagalur")
  ));

var boundary = district.geometry();
Map.centerObject(boundary, 10);
Map.addLayer(boundary, {color: "red"}, "Chikkamagaluru District");

// ── 2. Class remapping function ──────────────────────────────
// Unified classes:
// 1=Forest, 2=Shrubland, 3=Cropland, 4=Grassland,
// 5=Built-up, 6=Water/Wetland, 7=Bare/Snow

// ── 3. Year A — 2013 (MODIS MCD12Q1) ────────────────────────
// MODIS IGBP scheme → unified 7 classes
var modis_2013 = ee.ImageCollection("MODIS/061/MCD12Q1")
  .filterDate("2013-01-01", "2013-12-31")
  .first()
  .select("LC_Type1")
  .clip(boundary);

// IGBP remapping to unified classes
var modis_from = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17];
var modis_to   = [1,  1,  1,  1,  1,  2,  2,  2,  4,  4,  6,  3,  5,  3,  7,  7,  6];

var yearA = modis_2013.remap(modis_from, modis_to, 0)
  .rename("class")
  .uint8();

// Visualise Year A
var palette = ["#2d6a4f","#74c69d","#c9a227","#95d5b2","#bc4749","#4895ef","#d4a373"];
Map.addLayer(yearA, {min:1, max:7, palette: palette}, "Year A — 2013 (MODIS)");

// ── 4. Year B — 2022 (Dynamic World) ────────────────────────
// Dynamic World: 0=water,1=trees,2=grass,3=flooded_veg,
// 4=crops,5=shrub_scrub,6=built,7=bare,8=snow_ice
var dw_2022 = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
  .filterDate("2022-01-01", "2022-12-31")
  .filterBounds(boundary)
  .select("label")
  .mode()   // most frequent class per pixel across the year
  .clip(boundary);

var dw_from = [0, 1, 2, 3, 4, 5, 6, 7, 8];
var dw_to   = [6, 1, 4, 6, 3, 2, 5, 7, 7];

var yearB = dw_2022.remap(dw_from, dw_to, 0)
  .rename("class")
  .uint8();

Map.addLayer(yearB, {min:1, max:7, palette: palette}, "Year B — 2022 (Dynamic World)");

// ── 5. Legend ────────────────────────────────────────────────
var legend = ui.Panel({style:{position:"bottom-left",padding:"8px 12px"}});
legend.add(ui.Label("Land Cover Classes", {fontWeight:"bold", fontSize:"13px"}));
var classes = [
  ["1 — Forest",        "#2d6a4f"],
  ["2 — Shrubland",     "#74c69d"],
  ["3 — Cropland",      "#c9a227"],
  ["4 — Grassland",     "#95d5b2"],
  ["5 — Built-up",      "#bc4749"],
  ["6 — Water/Wetland", "#4895ef"],
  ["7 — Bare/Snow",     "#d4a373"],
];
classes.forEach(function(c){
  var row = ui.Panel({layout: ui.Panel.Layout.flow("horizontal")});
  row.add(ui.Label("", {backgroundColor:c[1], padding:"6px", margin:"2px 6px 2px 0"}));
  row.add(ui.Label(c[0], {margin:"2px 0"}));
  legend.add(row);
});
Map.add(legend);

// ── 6. Export Year A (2013 MODIS) to Google Drive ───────────
Export.image.toDrive({
  image:          yearA,
  description:    "Chikkamagaluru_2013_MODIS_classified",
  folder:         "GEE_LandUse",
  fileNamePrefix: "chikkamagaluru_2013",
  region:         boundary,
  scale:          30,            // 30 m (resampled from 500 m)
  crs:            "EPSG:32643",  // UTM Zone 43N — best for this region
  maxPixels:      1e13,
  fileFormat:     "GeoTIFF"
});

// ── 7. Export Year B (2022 Dynamic World) to Google Drive ───
Export.image.toDrive({
  image:          yearB,
  description:    "Chikkamagaluru_2022_DynamicWorld_classified",
  folder:         "GEE_LandUse",
  fileNamePrefix: "chikkamagaluru_2022",
  region:         boundary,
  scale:          30,            // 30 m native for Dynamic World
  crs:            "EPSG:32643",
  maxPixels:      1e13,
  fileFormat:     "GeoTIFF"
});

print("✓ Two export tasks submitted.");
print("Go to the Tasks tab (top-right) and click RUN on each task.");
print("Files will appear in your Google Drive under: GEE_LandUse/");
print("Year A: chikkamagaluru_2013.tif");
print("Year B: chikkamagaluru_2022.tif");
