#!/usr/bin/env python3
"""
Create a cell area raster (m²) from a reference raster.

Handles both projected (metric) and geographic (degree) CRS:
  - Projected CRS in metres: uses pixel dimensions directly (fast)
  - Geographic CRS (degrees): uses geodetic calculation via Geod (accurate but slower)

Usage:
    python create_cell_area_raster.py <reference_raster> <output_path>

Example:
    python create_cell_area_raster.py burkina_valid_mask.tif cell_area_m2.tif
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from pyproj import Geod
import copy


def compute_cell_area_projected(transform: Affine) -> float:
    """
    Compute pixel area for projected CRS (metres).
    
    Args:
        transform: rasterio Affine transform
    
    Returns:
        pixel area in m²
    """
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)
    return pixel_width * pixel_height


def compute_cell_area_geographic(transform: Affine, row: int, col: int, ellps: str = "WGS84") -> float:
    """
    Compute pixel area for geographic CRS (degrees) using geodetic calculation.
    
    Args:
        transform: rasterio Affine transform
        row: pixel row index
        col: pixel column index
        ellps: ellipsoid name for Geod (default: WGS84)
    
    Returns:
        pixel area in m²
    """
    geod = Geod(ellps=ellps)
    
    # Compute coordinates of pixel corners
    lon_ul, lat_ul = transform * (col, row)
    lon_ur, lat_ur = transform * (col + 1, row)
    lon_ll, lat_ll = transform * (col, row + 1)
    lon_lr, lat_lr = transform * (col + 1, row + 1)

    # Define pixel polygon (counter-clockwise)
    lons = [lon_ul, lon_ur, lon_lr, lon_ll, lon_ul]
    lats = [lat_ul, lat_ur, lat_lr, lat_ll, lat_ul]

    # Compute area in square meters
    area, _ = geod.polygon_area_perimeter(lons, lats)
    return abs(area)


def create_cell_area_raster(reference_path: Path, output_path: Path):
    """
    Read reference raster and create cell area raster in m².
    
    Args:
        reference_path: path to reference raster
        output_path: path to output cell area raster
    """
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference raster not found: {reference_path}")
    
    print(f"Reading reference raster: {reference_path}")
    
    with rasterio.open(reference_path) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        nodata_value = src.nodata if src.nodata is not None else -9999.0
        
        print(f"  Size: {height} x {width}")
        print(f"  CRS: {crs}")
        print(f"  Transform: {transform}")
    
    # Determine if CRS is projected or geographic
    is_projected = crs is not None and crs.is_projected
    
    print(f"  CRS type: {'projected (metres)' if is_projected else 'geographic (degrees)'}")
    
    # Compute cell areas
    print("Computing cell areas...")
    area_array = np.zeros((height, width), dtype=np.float32)
    
    if is_projected:
        # Fast path: constant pixel area
        pixel_area = compute_cell_area_projected(transform)
        area_array[:, :] = pixel_area
        print(f"  Pixel area (constant): {pixel_area:.2f} m²")
    else:
        # Geodetic path: per-pixel calculation
        if crs is None:
            warnings.warn(
                "CRS is None. Assuming WGS84 (EPSG:4326) for geodetic calculation. "
                "If this is incorrect, reproject to a projected CRS.",
                UserWarning,
            )
        
        print("  (geodetic calculation, may be slow for large rasters)")
        for row in range(height):
            if (row + 1) % max(1, height // 10) == 0:
                print(f"    Progress: {row + 1}/{height}")
            
            for col in range(width):
                area = compute_cell_area_geographic(transform, row, col)
                area_array[row, col] = area
    
    # Set nodata cells to nodata value
    print(f"Setting nodata cells to {nodata_value}")
    with rasterio.open(reference_path) as ref_src:
        ref_data = ref_src.read(1)
        ref_nodata = ref_src.nodata
        
        if ref_nodata is not None:
            area_array[ref_data == ref_nodata] = nodata_value
    
    # Update profile for output
    profile.update(
        driver="GTiff",
        dtype=rasterio.float32,
        nodata=nodata_value,
    )
    
    # Add metadata
    if "tags" not in profile:
        profile["tags"] = {}
    profile["tags"].update({
        "AREA_UNITS": "square_meters",
        "DESCRIPTION": "Cell area in m² (geodetic projection)" if not is_projected else "Cell area in m² (metric projection)",
        "CRS_TYPE": "geographic" if not is_projected else "projected",
    })
    
    # Write output
    print(f"Writing output raster: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(area_array, 1)
    
    print(f"Done. Cell area raster written to: {output_path}")
    print(f"  Min area: {np.nanmin(area_array[area_array != nodata_value]):.2f} m²")
    print(f"  Max area: {np.nanmax(area_array[area_array != nodata_value]):.2f} m²")
    print(f"  Mean area: {np.nanmean(area_array[area_array != nodata_value]):.2f} m²")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("ERROR: Missing arguments.")
        sys.exit(1)
    
    reference_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        create_cell_area_raster(reference_path, output_path)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
