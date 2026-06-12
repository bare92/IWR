from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS


@dataclass
class CropFraction:
    crop_type: list[str]
    raster: np.ndarray  # shape: (bands, rows, cols)
    nodata: float | int | None
    transform: Affine
    crs: CRS | None
    path: str

    @classmethod
    def from_geotiff(
        cls,
        geotiff_path: str,
        crop_type: Sequence[str] | None = None,
    ) -> "CropFraction":
        """Load a multiband fractional GeoTIFF into CropFraction.

        If crop_type is not provided, crop names are inferred from band descriptions.
        Example description: 'frac_wheat' -> crop type 'wheat'.
        """
        p = Path(geotiff_path)
        if not p.exists():
            raise FileNotFoundError(f"GeoTIFF not found: {geotiff_path}")

        with rasterio.open(p) as src:
            raster = src.read().astype(np.float32)
            band_count = src.count
            nodata = src.nodata
            transform = src.transform
            crs = src.crs

            desc = list(src.descriptions or [])
            desc = [d if d else f"band_{i+1}" for i, d in enumerate(desc)]

        inferred_crop_types = [
            d.replace("frac_", "", 1) if d.startswith("frac_") else d
            for d in desc
        ]

        if crop_type is None:
            crop_list = inferred_crop_types
        else:
            crop_list = [str(c) for c in crop_type]
            if len(crop_list) != band_count:
                raise ValueError(
                    f"crop_type length ({len(crop_list)}) must match number "
                    f"of bands ({band_count})."
                )

        return cls(
            crop_type=crop_list,
            raster=raster,
            nodata=nodata,
            transform=transform,
            crs=crs,
            path=str(p),
        )

    def get_crop_fraction(self, crop: str) -> np.ndarray:
        """Return the 2D fraction raster for a given crop name."""
        if crop not in self.crop_type:
            raise KeyError(f"Crop '{crop}' not found. Available: {self.crop_type}")
        idx = self.crop_type.index(crop)
        return self.raster[idx]


@dataclass
class SoilLayer:
    soil_type: np.ndarray  # shape: (rows, cols)
    nodata: float | int | None
    transform: Affine
    crs: CRS | None
    path: str

    @classmethod
    def from_geotiff(cls, geotiff_path: str) -> "SoilLayer":
        """Load a soil type GeoTIFF as a single-band raster."""
        p = Path(geotiff_path)
        if not p.exists():
            raise FileNotFoundError(f"GeoTIFF not found: {geotiff_path}")

        with rasterio.open(p) as src:
            arr = src.read(1)
            nodata = src.nodata
            transform = src.transform
            crs = src.crs

        return cls(
            soil_type=arr,
            nodata=nodata,
            transform=transform,
            crs=crs,
            path=str(p),
        )