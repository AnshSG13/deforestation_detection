"""
Deforestation‑mask creator – NumPy‑only output (GeoTIFF optional)

Mask values
-----------
0 : background / no deforestation
1 : Class 1 – target‑year deforestation
2 : Class 2 – earlier / accumulated deforestation
(Class 2 is burned **after** Class 1, so it overwrites.)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime  # retained for potential logging

# ── third‑party ────────────────────────────────────────────────────────────────
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import (
    box, Polygon, MultiPolygon, GeometryCollection
)
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Helper – keep only Polygonal geometry, coerce to MultiPolygon
# ──────────────────────────────────────────────────────────────────────────────
def _polygonize(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Return *gdf* containing only valid MultiPolygons.
    • Explodes GeometryCollections and drops non‑area parts
    • Wraps single Polygons into a MultiPolygon
    """
    if gdf.empty:
        return gdf

    def _to_multipolygon(geom):
        if not geom or geom.is_empty:
            return None

        # 1. GeometryCollection  → collect polygons
        if isinstance(geom, GeometryCollection):
            polys: list[Polygon] = []
            for g in geom.geoms:
                if isinstance(g, Polygon):
                    polys.append(g)
                elif isinstance(g, MultiPolygon):
                    polys.extend(list(g.geoms))
            return MultiPolygon(polys) if polys else None

        # 2. Plain Polygon       → wrap
        if isinstance(geom, Polygon):
            return MultiPolygon([geom])

        # 3. MultiPolygon        → keep
        if isinstance(geom, MultiPolygon):
            return geom

        # 4. Anything else       → drop
        return None

    out = gdf.copy()
    out["geometry"] = out.geometry.map(_to_multipolygon)
    out = out.dropna(subset=["geometry"])
    # set_geometry returns a new GDF when inplace=False (keeps CRS)
    return out.set_geometry("geometry", inplace=False)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: rectangle AOI
# ──────────────────────────────────────────────────────────────────────────────
def define_aoi(west, south, east, north, crs: str = "EPSG:4674") -> gpd.GeoDataFrame:
    """Return a GeoDataFrame representing the rectangular AOI in *crs*."""
    return gpd.GeoDataFrame({"geometry": [box(west, south, east, north)]}, crs=crs)


# ──────────────────────────────────────────────────────────────────────────────
# Main routine
# ──────────────────────────────────────────────────────────────────────────────
def create_mask_from_shapefiles(
    recent_defor_shapefile_path: str | Path,
    accumulated_defor_shapefile_path: str | Path,
    reference_vrt_path: str | Path,
    *,
    output_tiff_path: str | Path | None = None,
    output_npy_path: str | Path | None = None,
    target_year: int = 2023,
    date_field: str = "year",
    aoi_gdf: gpd.GeoDataFrame | None = None,
    location_name: str | None = None,
):
    """Create a single‑band deforestation mask and save as .npy (and GeoTIFF)."""

    # ── AOI from YAML (optional) ────────────────────────────────────────────
    if location_name:
        try:
            with open("aoi_config.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            coords = cfg[location_name]
            print(f"\nProcessing location: {location_name}")
            aoi_gdf = define_aoi(coords["west"], coords["south"],
                                 coords["east"], coords["north"])
            print("  AOI defined.")
        except Exception as e:
            print(f"Error loading AOI for '{location_name}': {e}")
            return

    # ── read shapefiles ────────────────────────────────────────────────────
    def _read_gdf(path):
        try:
            gdf = gpd.read_file(path)
            print(f"Loaded {len(gdf):,} features · CRS: {gdf.crs} · {Path(path).name}")
            return gdf
        except Exception as err:
            print(f"Error reading {path}: {err}")
            return None

    print("\nReading recent & accumulated shapefiles…")
    gdf_recent = _read_gdf(recent_defor_shapefile_path)
    gdf_accum  = _read_gdf(accumulated_defor_shapefile_path)
    if gdf_recent is None or gdf_accum is None:
        return

    # ── reference raster grid ───────────────────────────────────────────────
    try:
        with rasterio.open(reference_vrt_path) as src:
            transform = src.transform
            crs       = src.crs
            width, height = src.width, src.height
            ref_profile   = src.profile
        ref_profile["driver"] = "GTiff"          # needed only for GeoTIFF
        print(f"Reference grid: {width}×{height} · CRS: {crs}")
    except Exception as err:
        print(f"Error opening reference VRT: {err}")
        return

    # ── ensure date field is numeric ────────────────────────────────────────
    if date_field not in gdf_recent.columns:
        print(f"Column '{date_field}' not found in recent data.")
        return

    gdf_recent[date_field] = (
        pd.to_numeric(gdf_recent[date_field], errors="coerce")
          .fillna(pd.to_datetime(gdf_recent[date_field], errors="coerce").dt.year)
          .astype("Int64")
    )
    gdf_recent = gdf_recent.dropna(subset=[date_field])

    # ── split recent into target year / past ────────────────────────────────
    gdf_target       = gdf_recent[gdf_recent[date_field] == target_year].copy()
    gdf_recent_past  = gdf_recent[gdf_recent[date_field] < target_year].copy()
    print(f"Target‑year features: {len(gdf_target):,} · "
          f"Past (recent): {len(gdf_recent_past):,}")

    # ── combine past data (recent‑past + accumulated) ───────────────────────
    if not gdf_accum.empty and gdf_accum.crs != gdf_recent_past.crs:
        gdf_accum = gdf_accum.to_crs(gdf_recent_past.crs)

    gdf_past_all = gpd.GeoDataFrame(
        pd.concat([gdf_recent_past, gdf_accum], ignore_index=True),
        crs=gdf_recent_past.crs or gdf_accum.crs               # ⭐ keep CRS
    )
    print(f"Total combined past features: {len(gdf_past_all):,}")

    # ── clip to AOI (if provided) ───────────────────────────────────────────
    if aoi_gdf is not None and not aoi_gdf.empty:
        data_crs = gdf_target.crs or gdf_past_all.crs
        aoi_gdf  = aoi_gdf.to_crs(data_crs)
        if not gdf_target.empty:
            gdf_target = gpd.overlay(gdf_target, aoi_gdf,
                                     how="intersection", keep_geom_type=False)
        if not gdf_past_all.empty:
            gdf_past_all = gpd.overlay(gdf_past_all, aoi_gdf,
                                       how="intersection", keep_geom_type=False)
        print(f"After AOI: {len(gdf_target):,} target · {len(gdf_past_all):,} past")

    # ── reproject to raster CRS, then clean geometry ────────────────────────
    gdf_target   = gdf_target.to_crs(crs)   if not gdf_target.empty   else gdf_target
    gdf_past_all = gdf_past_all.to_crs(crs) if not gdf_past_all.empty else gdf_past_all

    gdf_target   = _polygonize(gdf_target)
    gdf_past_all = _polygonize(gdf_past_all)

    # ── initialise mask ─────────────────────────────────────────────────────
    mask = np.zeros((height, width), dtype=np.uint8)

    # ── 3‑pixel buffer (Class 2 ring) around Class 1 ────────────────────────
    if not gdf_target.empty:
        pixel_size = abs(transform.a)
        buf_dist   = pixel_size * 3

        gdf_buf = gdf_target.copy()
        gdf_buf.geometry = gdf_buf.geometry.buffer(buf_dist)

        # ring = buffer – original
        gdf_ring = gpd.overlay(gdf_buf, gdf_target, how="difference",
                               keep_geom_type=False)
        gdf_ring = _polygonize(gdf_ring)

        shapes_ring = ((geom, 2) for geom in gdf_ring.geometry if geom.is_valid)
        ring_ras = rasterize(
            shapes_ring, out_shape=(height, width), transform=transform,
            dtype=np.uint8, fill=0, default_value=2
        )
        new_buf_pixels = (ring_ras == 2) & (mask == 0)
        mask[new_buf_pixels] = 2
        print(f"  Buffer pixels (class 2): {int(new_buf_pixels.sum()):,}")

    # ── rasterise Class 1, then Class 2 ─────────────────────────────────────
    print("\nRasterising…")

    if not gdf_target.empty:
        shapes_1 = ((geom, 1) for geom in gdf_target.geometry if geom.is_valid)
        ras_1 = rasterize(
            shapes_1, out_shape=(height, width), transform=transform,
            dtype=np.uint8, fill=0, default_value=1
        )
        mask[ras_1 == 1] = 1
        print(f"  Pixels Class 1: {int((ras_1 == 1).sum()):,}")

    if not gdf_past_all.empty:
        shapes_2 = ((geom, 2) for geom in gdf_past_all.geometry if geom.is_valid)
        ras_2 = rasterize(
            shapes_2, out_shape=(height, width), transform=transform,
            dtype=np.uint8, fill=0, default_value=2
        )
        mask[ras_2 == 2] = 2
        print(f"  Pixels Class 2: {int((ras_2 == 2).sum()):,}")

    # ── diagnostics ─────────────────────────────────────────────────────────
    vals, counts = np.unique(mask, return_counts=True)
    print("Unique values:", dict(zip(vals, counts)))

    # ── save outputs ────────────────────────────────────────────────────────
    if output_tiff_path:
        ref_profile.update(dtype=mask.dtype, count=1, nodata=None)
        print(f"Saving GeoTIFF → {output_tiff_path}")
        with rasterio.open(output_tiff_path, "w", **ref_profile) as dst:
            dst.write(mask, 1)
        print("GeoTIFF saved.")

    if output_npy_path:
        print(f"Saving NumPy → {output_npy_path}")
        np.save(output_npy_path, mask)
        print("NumPy array saved.")

    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: total area of non‑zero mask
# ──────────────────────────────────────────────────────────────────────────────
def calculate_mask_area_v2(mask: np.ndarray, transform: rasterio.Affine) -> float:
    """Return total area (km²) of non‑zero pixels in *mask*."""
    pixel_area = abs(transform.a) * abs(transform.e)
    return mask.astype(bool).sum() * pixel_area / 1e6
