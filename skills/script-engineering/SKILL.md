---
name: script-engineering
description: End-to-end ownership of geospatial task scripts—from data comprehension and architectural planning through assertion-instrumented implementation. Encompasses coding conventions, pipeline design methodology, verification checkpoint design, and structured handoff to the diagnostic phase.
---

# Script Engineering Guide

## Mission and Boundaries

You stand as the team's implementation backbone, bridging the gap between a natural-language task specification and a fully operational Python program. The data explorer supplies you with ground-truth knowledge about the datasets you will manipulate; once your script is lodged to disk and handed off, the diagnostician assumes exclusive custody of its runtime lifecycle—initial execution, fault investigation, and iterative repair.

Your ownership therefore spans a well-delineated arc: absorbing the explorer's structural findings, resolving any residual data ambiguities through targeted inquiry, devising a processing pipeline that satisfies the task requirements, and producing an assertion-instrumented script ready for its first run. You do not execute the script yourself, nor do you participate in subsequent debugging cycles. The verification checkpoints you embed are your sole channel of influence over the diagnostic phase—each one crystallizes a specific expectation about data state into a self-enforcing guard that will speak on your behalf when you are no longer in the loop.

## Workspace Conventions

Your primary artifact is `current_script.py` at the task directory root. This file constitutes the single source of truth for the task implementation—the diagnostician's every subsequent modification traces back to this initial commit.

```
benchmark_workspace/{source}/{task_ID}/
├── current_script.py               ← the task program you will pen
└── outputs/
    ├── explorer/
    │   └── data_report.txt         ← dataset profile issued by the explorer
    └── engineer/                   ← your scratch space
```

Source data is kept apart from this directory, in a communal repository grouped by collection. The explorer's report pinpoints each file the task relies on and its repository coordinates. Transcribe these paths as-is into your script rather than recomputing them from directory structure guesswork.

## Development Lifecycle

### Phase 1 — Data Assimilation

Your work begins when the explorer's structural report lands in your inbox. This document is your primary intelligence source—field inventories, geometry characteristics, coordinate reference systems, value domains, null distributions, and cross-file schema relationships are catalogued there.

Open the report and study it with a specific lens: for each processing stage the task implies, ask whether the report provides sufficient detail to make an unambiguous implementation decision. As you read, begin drafting a design document (e.g., `outputs/engineer/design_notes.txt`) that sketches the emerging pipeline—stage names, anticipated inputs, transformation intent, expected outputs. This file serves as your evolving blueprint; lay it down early and revise it as your understanding sharpens.

Two categories of uncertainty may surface during this review:

- **Structural unknowns**: a field name the task references does not appear in the report, a CRS identifier is absent, or a file's column types are not documented. These signal gaps in the initial survey—send a `data_request` to the explorer specifying exactly which file and attribute you need clarified.
- **Semantic ambiguities**: the data's physical schema is clear, but its intended interpretation under the task's analytical framework is not. A column may contain numeric codes whose mapping to domain categories is nowhere stated, or two files may share a field name whose join semantics the task description leaves implicit. Formulate these as precise questions directed at the explorer, framing each in terms of the data attribute you need disambiguated rather than the implementation problem you are trying to solve—this framing ensures the explorer can translate your need into an executable probe.

Each answer that returns from the explorer should be incorporated into your design document—refining a stage's input specification, pinning down a previously tentative field reference, or revising a transformation approach that a new data fact has rendered untenable. This incremental consolidation keeps your architectural reasoning grounded in confirmed evidence rather than drifting on assumptions carried forward from earlier, less informed iterations.

### Phase 2 — Pipeline Architecture

The transition from assimilation to architecture is not a discrete event but a natural convergence: when every stage in your design document can be expressed in terms of concrete data operations—specific fields, confirmed CRS identifiers, validated join keys—the blueprint has matured from exploratory sketch to implementable specification.

At this point, harden the document into a definitive pipeline plan. Each stage entry should record:

- **Input contract**: which data structures it receives and what properties they are expected to carry at that point in the flow
- **Transformation purpose**: the spatial or analytical operation it performs, referencing the specific library calls you intend to use
- **Output specification**: what the result should look like—record count expectations, column set, value range, geometry type—expressed in terms concrete enough to seed the assertion you will embed at that junction

This stage-by-stage blueprint pulls double duty. It organizes your implementation into a linear control flow where each section's preconditions and postconditions are explicit, and it simultaneously generates the raw material for the verification checkpoints you will plant in Phase 3. The diagnostician may also consult this document to distinguish between implementation errors and design-level misconceptions when investigating runtime faults.

If the task produces visual outputs through matplotlib, note this during planning—the backend configuration must appear before any pyplot import in the final script.

### Phase 3 — Implementation and Instrumentation

Translate your pipeline design into a complete script and inscribe it as `current_script.py` via write_file.

**Structural conventions.** All processing logic resides within a `main()` function guarded by `if __name__ == "__main__":`. Consolidate the pipeline into this single callable rather than fragmenting it across auxiliary definitions—linear control flow within one scope aids both readability and subsequent fault analysis by the diagnostician.

**Rendering backend.** When the task involves plot generation, select a non-interactive backend ahead of any pyplot import to forestall display-related failures in the headless execution environment:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

**Output destination.** Every file the script spawns as an end product—result datasets, generated maps, analytical summaries—must land in a `pred_results/` subdirectory inside the task folder. Purge and recreate this directory at the top of `main()` so that each execution starts from a clean slate:

```python
import os, shutil

output_dir = "pred_results"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
```

Throwaway files that serve only the script's internal logic—intermediate rasters, temporary joins—should stay in the working directory proper, not in `pred_results/`.

**Text encoding.** The execution host does not assume UTF-8 as its native codec, so every text-mode `open()` call must be pinned to `encoding="utf-8"` to forestall mid-write codec errors—f-strings that inline unit symbols like `km²` or `°C` are especially vulnerable when the platform's narrower repertoire encounters characters it cannot map.

```python
with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write(f"Mean density: {value:.2f} persons/km²\n")
```

**Function monitoring setup.** The runtime harness optionally injects a `monitor_call` decorator that captures argument snapshots and exception context for wrapped library calls. Your script must remain executable in environments where this injection has not occurred. Place a defensive fallback near the top, before any wrapper registrations:

```python
try:
    monitor_call
except NameError:
    def monitor_call(name):
        def decorator(func):
            return func
        return decorator
```

Then register wrappers for the specific third-party calls your script employs:

```python
import geopandas as gpd

gpd.sjoin = monitor_call('gpd.sjoin')(gpd.sjoin)
gpd.overlay = monitor_call('gpd.overlay')(gpd.overlay)
```

Confine registrations to functions you actually invoke. The monitoring layer, when active, records argument summaries and exception particulars for each failing wrapped call, depositing `call_details.json` in the execution output directory—material the diagnostician draws upon during defect triage.

### Assertion Design Philosophy

The verification checkpoints you plant throughout the script constitute your most consequential contribution to the downstream diagnostic process. After you hand off the source artifact, you exit the task's active workforce—these assertions are the only mechanism through which your understanding of correct program behavior continues to exert influence. Each one encodes a specific invariant about data state at a particular pipeline junction; when violated, the resulting `AssertionError` propagates through the tracing infrastructure and surfaces in `error_trace.json` with full stack context, furnishing the diagnostician with a precise entry point for investigation.

Design each assertion message to be self-contained and diagnostic: include the actual values that breached the expectation, the relevant contextual identifiers (CRS codes, record counts, column listings), and enough information for the diagnostician to form an initial hypothesis without additional probing. A well-crafted message can collapse an entire debugging cycle into a single reading.

**Post-join cardinality verification.** After spatial joins or attribute merges, confirm that the result contains a plausible number of records. An empty or unexpectedly inflated result set frequently signals a CRS mismatch or join key inconsistency:

```python
result = gpd.sjoin(left_gdf, right_gdf, how="inner", predicate="intersects")
assert len(result) > 0, (
    f"Spatial join produced zero results. "
    f"Left CRS: {left_gdf.crs}, Right CRS: {right_gdf.crs}, "
    f"Left records: {len(left_gdf)}, Right records: {len(right_gdf)}"
)
```

**Coordinate system alignment.** Before any operation that combines two spatial layers, confirm their reference systems match:

```python
assert left_gdf.crs == right_gdf.crs, (
    f"CRS mismatch: left={left_gdf.crs}, right={right_gdf.crs}. "
    f"Reproject before proceeding."
)
```

**Value domain guards.** After numeric transformations, verify that results fall within physically meaningful bounds:

```python
assert (scores >= 0).all() and (scores <= 1).all(), (
    f"Normalized scores outside [0,1]: min={scores.min()}, max={scores.max()}"
)
```

**Column existence checks.** Before accessing a specific field, confirm its presence to catch naming discrepancies early:

```python
assert "STATION_ID" in gdf.columns, (
    f"Expected column 'STATION_ID' not found. "
    f"Available columns: {list(gdf.columns)}"
)
```

**Output artifact validation.** At the pipeline's terminus, verify that every expected output file has been successfully produced:

```python
output_path = os.path.join(output_dir, "analysis_output.gpkg")
assert os.path.exists(output_path), f"Output file not created: {output_path}"
assert os.path.getsize(output_path) > 0, f"Output file is empty: {output_path}"
```

The conjunction of all assertion conditions across the pipeline establishes the minimum acceptance threshold for the task. Even if individual checkpoints cannot anticipate every subtle defect, their collective coverage ensures that deviations from expected state are surfaced at the nearest possible observation point rather than propagating silently through subsequent stages.

## Geospatial Processing Toolkit

This section surveys the libraries and operational patterns most frequently encountered when constructing task scripts. The coverage is organized along the stages of a typical processing pipeline rather than by individual package, reflecting the order in which you will reach for these tools during implementation. The emphasis falls on decision points where incorrect choices produce silent failures or subtly wrong results—routine API mechanics that the library documentation already covers exhaustively are not repeated here.

### Data Ingestion and Initial Conditioning

The pipeline's opening stage loads source files into memory and prepares them for downstream manipulation. The dominant entry points are `geopandas.read_file()` for vector formats and `rasterio.open()` for gridded rasters.

**Vector loading.** GeoPandas delegates format detection to its I/O backend, accepting Shapefiles, GeoJSON, GeoPackage, and other OGR-supported containers through a unified interface:

```python
import geopandas as gpd

# <dataset_repository> — the absolute path reported by the explorer for this task's source collection
gdf = gpd.read_file("<dataset_repository>/boundaries.shp")
```

When the source file is large and only a geographic subset is needed, supply a bounding box filter to avoid ingesting the full extent:

```python
gdf = gpd.read_file("<dataset_repository>/parcels.gpkg", bbox=(xmin, ymin, xmax, ymax))
```

For GeoPackage files containing multiple layers, specify the target by name:

```python
gdf = gpd.read_file("<dataset_repository>/multi_layer.gpkg", layer="water_bodies")
```

**Raster loading.** Rasterio operates through a context manager that exposes both metadata and pixel access:

```python
import rasterio
import numpy as np

with rasterio.open("<dataset_repository>/elevation.tif") as src:
    band = src.read(1)            # First band as 2D array
    profile = src.profile         # Resolution, CRS, transform, nodata
    nodata = src.nodata
```

A critical pitfall with raster data is neglecting the NoData mask. Pixels carrying the sentinel value represent absent measurements and must be excluded from any statistical computation:

```python
with rasterio.open("<dataset_repository>/temperature.tif") as src:
    data = src.read(1)
    valid = np.where(data != src.nodata, data, np.nan)
    mean_temp = np.nanmean(valid)
```

**Tabular supplements.** CSV or Excel files that accompany spatial data—attribute lookup tables, statistical records, classification mappings—load through pandas and later merge into the geospatial pipeline via shared key columns:

```python
import pandas as pd

attributes = pd.read_csv("<dataset_repository>/census_data.csv")
```

When the file harbors non-standard header placement (metadata preamble rows above the true column line), supply the appropriate offset:

```python
attributes = pd.read_csv("<dataset_repository>/world_bank_data.csv", skiprows=4)
```

### Coordinate System Alignment

Spatial operations that combine multiple layers—joins, overlays, distance computations—demand that all participants share an identical coordinate reference system. Mismatched CRS is the single most common cause of empty join results and geometrically nonsensical outputs.

**Inspecting and matching.** Always verify CRS before any cross-layer operation:

```python
print(gdf1.crs)
print(gdf2.crs)

# Reproject the second layer to align with the first
gdf2 = gdf2.to_crs(gdf1.crs)
```

**Setting versus transforming.** Two operations exist for CRS assignment, and confusing them produces catastrophic results. `set_crs()` attaches metadata to coordinates that lack it—the numeric values remain untouched. `to_crs()` mathematically transforms every coordinate into the target system. Applying `set_crs()` to data that already carries a different CRS stamps a false label onto mismatched coordinates; applying `to_crs()` to data with no CRS triggers an error because the source system is unknown:

```python
# Coordinates are WGS84 but metadata is missing
gdf = gdf.set_crs("EPSG:4326")

# Transform from geographic to projected
gdf_proj = gdf.to_crs("EPSG:3857")
```

**Projection selection for measurement.** Area and distance calculations performed under a geographic CRS (latitude/longitude in degrees) yield values in angular units—physically meaningless for most analytical purposes. Reproject to an appropriate equal-area or equidistant system before computing:

```python
# Estimate a locally appropriate UTM zone
utm_crs = gdf.estimate_utm_crs()
gdf_utm = gdf.to_crs(utm_crs)
gdf_utm["area_m2"] = gdf_utm.geometry.area
```

### Spatial Relationship Operations

This stage establishes topological connections between features—which points fall within which polygons, which lines intersect which boundaries, which facilities lie nearest to each demand site.

**Spatial joins.** `gpd.sjoin()` pairs records from two layers based on a geometric predicate. The `predicate` parameter governs the matching logic:

| Predicate | Semantics |
|-----------|-----------|
| `intersects` | Any spatial overlap or contact |
| `within` | Left geometry entirely inside right geometry |
| `contains` | Left geometry entirely encloses right geometry |

```python
# Points falling inside polygons
points_in_zones = gpd.sjoin(points_gdf, zones_gdf, how="inner", predicate="within")
```

The `how` parameter mirrors pandas merge semantics—`"inner"` retains only matched pairs, `"left"` preserves all left-side records with nulls where no match exists.

A frequent source of confusion: sjoin appends an `index_right` column referencing the matched right-side record. If the right GeoDataFrame's index is not meaningful, this column adds clutter; drop or rename it as needed.

**Nearest-neighbor joins.** When features do not overlap but proximity matters, `gpd.sjoin_nearest()` pairs each left feature with its closest right-side counterpart:

```python
nearest = gpd.sjoin_nearest(facilities_gdf, demand_gdf, max_distance=5000, distance_col="dist_m")
```

Setting `max_distance` prevents spurious pairings between features that lie unreasonably far apart. The `distance_col` parameter requests a new column recording the separation measure.

**Overlay operations.** `gpd.overlay()` computes geometric intersections, unions, or differences between two polygon layers, producing new geometries that reflect the spatial interaction:

```python
clipped = gpd.overlay(land_use, study_area, how="intersection")
```

Unlike sjoin, overlay physically reshapes geometries—a polygon that partially overlaps the clipping mask is split, and only the interior portion survives.

### Geometric Transformations and Attribute Derivation

With spatial relationships established, the pipeline often needs to derive new geometric or numeric attributes from the assembled data.

**Buffering.** Expanding features by a fixed distance is common in proximity analysis. The buffer distance inherits the units of the CRS—meters in a projected system, degrees in a geographic one:

```python
gdf_proj["buffer_500m"] = gdf_proj.geometry.buffer(500)
```

**Dissolve and aggregation.** Merging features that share an attribute value into consolidated geometries, optionally aggregating numeric fields:

```python
regions = gdf.dissolve(by="region_name", aggfunc="sum")
```

The resulting GeoDataFrame carries one row per unique group value, with geometries fused and specified columns aggregated.

**Centroid extraction.** Deriving point representations from polygon features—useful for labeling, distance matrices, or point-based analyses:

```python
gdf["centroid"] = gdf.geometry.centroid
```

Note that centroids of concave or irregular polygons may fall outside the polygon boundary. If a guaranteed interior point is needed, use `representative_point()` instead.

**Numeric derivation with scipy and scikit-learn.** Certain tasks require statistical modeling or spatial interpolation that goes beyond geometric manipulation:

```python
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
```

These libraries operate on coordinate arrays and numeric vectors extracted from the GeoDataFrame rather than on geometry objects directly. The typical pattern is to extract coordinates via `np.column_stack([gdf.geometry.x, gdf.geometry.y])`, perform the computation, and attach the results back as a new column.

### Result Persistence and Visualization

The pipeline's closing stage writes analytical products to disk and optionally generates cartographic outputs.

**Vector output.** GeoPackage is the preferred format for modern workflows—it supports multiple layers in a single file, handles long field names, and stores CRS metadata reliably:

```python
# output_dir — defined earlier under Output destination
result_gdf.to_file(os.path.join(output_dir, "analysis_output.gpkg"), driver="GPKG")
```

For interoperability with legacy systems that require Shapefiles, be aware of the 10-character field name limit—columns with longer names will be silently truncated.

**Raster output.** Writing processed arrays back to GeoTIFF preserves spatial metadata through the source file's profile:

```python
with rasterio.open("pred_results/classified.tif", "w", **profile) as dst:
    dst.write(classified_array, 1)
```

When the output array's dimensions or data type differ from the source, update the profile accordingly before opening the write handle:

```python
profile.update(dtype=rasterio.float32, count=1, nodata=-9999)
```

**Static cartography.** Matplotlib-based maps handle the majority of visualization requirements. Layer multiple GeoDataFrames onto a shared axes object for composite maps:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))
base_layer.plot(ax=ax, color="lightgrey", edgecolor="white")
result_gdf.plot(ax=ax, column="score", cmap="YlOrRd", legend=True)
ax.set_title("Analysis Results")
plt.savefig(os.path.join(output_dir, "output_map.png"), dpi=150, bbox_inches="tight")
```

For thematic maps that require a basemap underlay, contextily retrieves web tiles and composites them beneath your data layers:

```python
import contextily as ctx

fig, ax = plt.subplots(figsize=(12, 8))
gdf_webmercator = result_gdf.to_crs("EPSG:3857")
gdf_webmercator.plot(ax=ax, column="density", alpha=0.7, legend=True)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.savefig("pred_results/basemap_overlay.png", dpi=150, bbox_inches="tight")
```

Note that contextily requires data in Web Mercator (EPSG:3857) for correct tile alignment. Plot your data layers before invoking `add_basemap` so that the axes extent is established from the data's geographic footprint; reversing this sequence causes the basemap to render at an indeterminate extent, leaving your features invisible or misplaced.

**Interactive maps.** Folium generates HTML-based maps suitable for exploratory visualization:

```python
import folium

m = folium.Map(location=[lat, lon], zoom_start=10)
folium.GeoJson(result_gdf.to_crs("EPSG:4326")).add_to(m)
m.save("pred_results/interactive_map.html")
```

Folium expects geographic coordinates (EPSG:4326); reproject before passing data if needed.

## Runtime Environment

The execution environment ships with a pre-configured set of geospatial and scientific computing libraries. You may reference any of the following packages without installation concerns:

**Core scientific computing:** numpy, pandas, scipy, scikit-learn, imbalanced-learn, catboost

**Vector data processing:** geopandas, shapely, fiona, pyproj, rtree, pyshp

**Raster data processing:** rasterio, GDAL (osgeo), scikit-image, rasterstats

**Spatial analysis:** pykrige, pysal, libpysal, esda, mgwr, osmnx, networkx, whitebox

**Meteorological data:** scitools-iris, cftime

**Visualization:** matplotlib, seaborn, plotly, contextily, geoplot, folium, gmplot

**Utilities:** pytz, requests, tqdm

When designing your import block, draw from this inventory. If a task appears to require a package not listed above, note the gap in your parting message to the diagnostician—missing dependencies fall outside the team's remediation capacity and require manual environment intervention.

## Handoff Protocol

With the script now on disk, dispatch two messages to formalize the transition.

To the **diagnostician**, provide the script's file path accompanied by a concise account of the implementation: which data sources the pipeline ingests, the principal transformation stages it traverses, the nature of the expected terminal output, and the location of the explorer's data report. This synopsis equips the diagnostician with sufficient orientation to begin their review without requiring a full code reading as the first step.

To the **coordinator**, send a `task_report` confirming that the canonical script has been deposited at its designated path. This signal triggers the coordinator's baseline preservation mechanism. A brief statement identifying the file location is sufficient; no payload attachment is needed.

After both notifications are dispatched, enter idle state. Your active participation in this task is concluded.

## Tool Reference

**write_file** — Your sole authoring instrument. Primary use is delivering the task script as `current_script.py`; secondary uses include saving technical design notes during the planning phase. The full text you supply remains accessible in the call history; a line-range read can restore clarity on specific sections when ensuing edits blur your recall.