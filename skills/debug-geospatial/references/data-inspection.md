# Data Inspection Reference

## When to Consult This Guide

Turn to this document when debugging evidence points beyond code logic toward the data itself. Typical triggers include:

- A `KeyError` or `ValueError` referencing a column name that appears in the task prompt but not in the actual dataset
- Spatial operations returning empty results despite seemingly valid inputs
- Coordinate values falling outside expected geographic bounds
- Silent dtype coercion producing nonsensical computed outputs
- Discrepancies between the number of records the prompt describes and what the script encounters at runtime

The benchmark dataset exhibits documented naming divergence between task descriptions and physical files at the filename level. Similar inconsistencies may extend to field names within datasets—casing, abbreviation, or pluralization differences—though this remains an empirical possibility rather than a confirmed pattern. When column-level errors surface, verifying field alignment against the actual data should precede algorithmic investigation.

## Workspace Data Layout

The task directory structure is outlined in the main skill guide. Within that layout, the `dataset/` subdirectory contains every geospatial resource the script is expected to consume. File names within this directory are authoritative—when they conflict with the task prompt's description, the physical file system takes precedence.

To enumerate available resources before diving into format-specific inspection, execute a directory listing through `run_utility_script`:

```python
import os
for f in sorted(os.listdir('dataset')):
    print(f)
```

## Format-Specific Inspection

### Vector Data (Shapefile, GeoJSON, GeoPackage)

Vector datasets encode discrete geographic features—points, lines, polygons—alongside tabular attributes. Key dimensions to examine:

**Field inventory**: Column names and their data types constitute the most frequent source of mismatch errors. Pay particular attention to abbreviations imposed by the Shapefile format's 10-character field name limit, which often truncates descriptive labels found in the task prompt.

**Geometry characteristics**: The distribution of geometry types (Point, MultiPoint, Polygon, MultiPolygon) determines which spatial operations are applicable. Mixed-type collections can cause unexpected failures in functions that assume homogeneous geometry.

**Coordinate reference system**: Mismatched CRS between layers is a pervasive cause of empty spatial joins and erroneous distance calculations. Verify that all datasets participating in a spatial operation share the same reference frame, or that appropriate reprojection occurs beforehand.

**Record count and null distribution**: An unexpectedly low record count may indicate a filtering issue upstream, while concentrated null values in key columns can silently derail aggregation logic.

A pre-built inspection script covers all these dimensions in a single invocation:

```
run_utility_script(
    script_path="<absolute_path>/skills/debug-geospatial/scripts/inspect_vector.py",
    args=["dataset/filename.geojson"]
)
```

### Raster Data (GeoTIFF, IMG)

Raster datasets represent continuous spatial phenomena as gridded arrays of pixel values. Critical aspects to verify:

**Band composition**: Multi-band files (e.g., satellite imagery) pack several measurement channels into a single file. Band count and ordering may not match the task description—preprocessing steps sometimes drop or reorder bands without updating documentation. The inspection script reports band descriptions when embedded in the file metadata, enabling direct correlation with the expected channel lineup.

**Value domain**: Raw digital numbers, calibrated reflectance values, and classified categorical codes occupy vastly different numeric ranges. Code that applies arithmetic designed for one representation to another produces silently wrong results. Cross-reference the observed value range and data type against the task's stated data semantics.

**NoData handling**: Pixels marked as NoData represent absent or invalid measurements. If the NoData sentinel overlaps with legitimate data values, or if no sentinel is defined when one should be, masking operations will either exclude valid observations or incorporate garbage values.

**Spatial resolution and extent**: Pixel dimensions govern the geographic area each cell represents. Operations combining rasters of differing resolutions require explicit resampling, and misaligned extents lead to clipped or padded outputs.

```
run_utility_script(
    script_path="<absolute_path>/skills/debug-geospatial/scripts/inspect_raster.py",
    args=["dataset/filename.tif"]
)
```

### Tabular Data (CSV, Excel)

Plain tabular files serve as attribute sources, lookup tables, or intermediate processing inputs. Relevant inspection targets:

**Column semantics**: Header names in CSV files are sensitive to whitespace padding, encoding artifacts, and inconsistent quoting. A column that appears identical in the task description may carry a trailing space or BOM prefix that defeats exact string matching.

**Type inference pitfalls**: Pandas infers column types from content, which can misclassify numeric identifiers as integers (stripping leading zeros), interpret date strings as generic objects, or coerce mixed-type columns to object dtype. Verify that inferred types align with the intended semantics, especially for join keys and grouping variables.

**Missing value patterns**: The distribution of nulls across columns—whether sparse and random or concentrated in specific segments—affects aggregation outcomes and merge behavior. Columns with high null rates that serve as join keys will silently drop records during inner joins.

```
run_utility_script(
    script_path="<absolute_path>/skills/debug-geospatial/scripts/inspect_tabular.py",
    args=["dataset/filename.csv"]
)
```

## Pre-built Inspection Scripts

Three diagnostic routines reside in the skill's `scripts/` directory, each targeting a major format family:

| Script | Supported Formats | Output Highlights |
|--------|-------------------|-------------------|
| `inspect_vector.py` | .shp, .geojson, .gpkg | Field catalog, geometry type census, CRS, bounding envelope, null tally, sample records |
| `inspect_raster.py` | .tif, .img | Band inventory with descriptions, pixel dimensions, value statistics per channel, NoData configuration |
| `inspect_tabular.py` | .csv, .xls, .xlsx | Column types with uniqueness and null counts, row sample, numeric distribution summary |

All scripts accept a single positional argument—the target file path relative to the working directory—and emit structured plain text to stdout. Invoke them through `run_utility_script` with the `script_path` and `args` parameters.

## Ad-hoc Inspection Strategies

When the pre-built scripts do not cover a specific need, compose a targeted probe using `run_utility_script` with inline code. Common scenarios include:

**Selective field deep-dive**: Extract unique values or frequency distributions for a specific column suspected of containing unexpected entries.

**Cross-file schema comparison**: Load two datasets and diff their column sets to identify naming inconsistencies that would cause a join to fail silently.

**Geometry validity audit**: Iterate over features and flag invalid geometries using Shapely's `is_valid` predicate, reporting the nature and location of each defect.

**CRS compatibility check**: Load multiple layers and compare their coordinate reference systems, highlighting any that require reprojection before spatial operations.

**Encoding diagnosis**: When a CSV fails to parse or displays garbled characters, attempt loading with alternative encodings (`latin-1`, `cp1252`) to isolate the correct character set.

These patterns serve as starting points. Adapt the inspection logic to the specific evidence gathered from earlier debugging steps, rather than applying a fixed checklist indiscriminately.