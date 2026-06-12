
from iwr_processing.iwr_layers import CropFraction, SoilLayer


class IWRModel:
    def __init__(
        self,
        time_range: str | None,
        land_cover_path: str,
        soil_path: str,
    ):
        self.time_range = time_range
        self.crop_fractions = CropFraction.from_geotiff(land_cover_path)
        self.soil = SoilLayer.from_geotiff(soil_path)

    def summary(self) -> dict:
        return {
            "time_range": self.time_range,
            "land_cover_path": self.crop_fractions.path,
            "land_cover_bands": len(self.crop_fractions.crop_type),
            "crop_type": self.crop_fractions.crop_type,
            "soil_path": self.soil.path,
            "soil_shape": list(self.soil.soil_type.shape),
        }