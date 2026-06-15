
import numpy as np

from iwr_processing.iwr_layers import CropFraction, SoilLayer


def compute_taw(
    theta_fc: np.ndarray,
    theta_wp: np.ndarray,
    root_depth_m: float,
) -> np.ndarray:
    """Total available water in mm."""
    return 1000.0 * (theta_fc - theta_wp) * root_depth_m


def compute_raw(taw: np.ndarray, p: float) -> np.ndarray:
    """Readily available water in mm."""
    return p * taw


class IWRModel:
    def __init__(
        self,
        time_range: str | None,
        land_cover_path: str,
        soil_path: str,
        crop_parameter_csv: str | None = None,
    ):
        self.time_range = time_range
        self.crop_fractions = CropFraction.from_geotiff(
            land_cover_path,
            crop_parameter_csv=crop_parameter_csv,
        )
        self.soil = SoilLayer.from_geotiff(soil_path)

    def get_crop_taw(self, crop: str, root_depth_use: str = "max") -> np.ndarray:
        """Return TAW raster for one crop using soil FC/WP and crop root depth."""
        root_depth_m = self.crop_fractions.get_crop_root_depth(crop, use=root_depth_use)
        taw = compute_taw(
            self.soil.field_capacity,
            self.soil.wilting_point,
            root_depth_m,
        ).astype(np.float32)
        taw[~self.soil.hydraulic_mask()] = self.soil.hydraulic_nodata
        return taw

    def get_crop_raw(
        self,
        crop: str,
        etc_mm_day: float | None = None,
        root_depth_use: str = "max",
    ) -> np.ndarray:
        """Return RAW raster for one crop using FAO-56 p and TAW."""
        taw = self.get_crop_taw(crop, root_depth_use=root_depth_use)
        p_value = self.crop_fractions.get_crop_p(crop, etc_mm_day=etc_mm_day)
        raw = compute_raw(taw, p_value).astype(np.float32)
        raw[~self.soil.hydraulic_mask()] = self.soil.hydraulic_nodata
        return raw



    def summary(self) -> dict:
        return {
            "time_range": self.time_range,
            "land_cover_path": self.crop_fractions.path,
            "land_cover_bands": len(self.crop_fractions.crop_type),
            "crop_type": self.crop_fractions.crop_type,
            "has_crop_parameters": self.crop_fractions.crop_parameters is not None,
            "soil_path": self.soil.path,
            "soil_shape": list(self.soil.soil_type.shape),
            "soil_field_capacity_shape": list(self.soil.field_capacity.shape),
            "soil_wilting_point_shape": list(self.soil.wilting_point.shape),
            "soil_hydraulic_nodata": self.soil.hydraulic_nodata,
        }