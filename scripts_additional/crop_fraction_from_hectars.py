

import numpy as np
import glob
import pandas as pd
import rasterio




CROP_LAYERS_PATH = "/home/fremen/data/basedata/BURKINA_FASO/CROPGRIDS/tif"

OUTPUT_PATH = "/home/fremen/data/basedata/BURKINA_FASO/CROPGRIDS"

PIXEL_AREA_LAYER = "/home/fremen/data/projects/Burkina/00_Data_iwr/static/burkina_cell_area_m2.tif"

crop_layers = glob.glob(f"{CROP_LAYERS_PATH}/CROPG*.tif")

# save in a list the crops names extracted from the crop_layers list
crop_names = [crop_layer.split("/")[-1].split("_")[-2] for crop_layer in crop_layers]

# read the pixel area layer to get the pixel area values
with rasterio.open(PIXEL_AREA_LAYER) as src:
    pixel_area = src.read(1)

# allocate a empty numpy array to hold the data for all crop layers of data.shape rows and columns and len(crop_layers) as depth
with rasterio.open(crop_layers[0]) as src:
    data_shape = src.read(1).shape
data_fractional = np.full((len(crop_layers), data_shape[0], data_shape[1]), np.nan)
for crop_idx, crop_layer in enumerate(crop_layers):
    crop_name = crop_layer.split("/")[-1].split(".")[0]
    print(f"Processing {crop_name} ...")
    with rasterio.open(crop_layer) as src:
        data = src.read(1).astype(np.float64)
    valid = ~np.isnan(data) & (pixel_area > 0)
    data_fractional[crop_idx][valid] = ((data[valid] * 10000.0) / pixel_area[valid])*100
    print(f"Finished processing {crop_name}")
        

# save the data_fractional array to a new GeoTIFF file containing all fraction layers as separate bands
output_file = f"{OUTPUT_PATH}/CROPG_fractional.tif"
with rasterio.open(crop_layers[0]) as src:
    profile = src.profile
    profile.update(
        driver='GTiff',
        dtype=rasterio.float64,
        count=len(crop_layers),
        compress='lzw'
    )
    with rasterio.open(output_file, 'w', **profile) as dst:
        for crop_idx in range(len(crop_layers)):
            dst.write(data_fractional[crop_idx], crop_idx + 1)

# in pixel exceeding 90% of sum of crop fraction decrease their value by 5 %
# compute the sum of all crop fractions and check if it exceeds 100% in any pixel
total_fraction = np.nansum(data_fractional, axis=0)
exceeding_90 = total_fraction > 90
data_fractional[:, exceeding_90] *= 0.95

# compute the sum of all crop fractions and check if it exceeds 100% in any pixel
total_fraction = np.nansum(data_fractional, axis=0)
if np.any(total_fraction > 100):
    print("Warning: Total crop fraction exceeds 100% in some pixels.")
# save the total_fraction array to a new GeoTIFF file and a quality mask qith 1 where total_fraction > 100 and 0 otherwise
total_fraction_file = f"{OUTPUT_PATH}/CROPG_total_fraction.tif"
with rasterio.open(crop_layers[0]) as src:
    profile = src.profile
    profile.update(
        driver='GTiff',
        dtype=rasterio.float64,
        count=1,
        compress='lzw'
    )
    with rasterio.open(total_fraction_file, 'w', **profile) as dst:
        dst.write(total_fraction, 1)

# save a quality mask: 1 where total fraction > 100, 0 elsewhere
quality_mask = (total_fraction > 100).astype(np.uint8)
quality_mask_file = f"{OUTPUT_PATH}/CROPG_quality_mask.tif"
with rasterio.open(crop_layers[0]) as src:
    qprofile = src.profile
    qprofile.update(
        driver='GTiff',
        dtype=rasterio.uint8,
        count=1,
        compress='lzw',
        nodata=None,
    )
with rasterio.open(quality_mask_file, 'w', **qprofile) as dst:
    dst.write(quality_mask, 1)
print(f"Saved quality mask: {quality_mask_file}  (flagged pixels: {int(quality_mask.sum())})")

# save a csv with the crop names and their corresponding 
# band index in the output file under the column crop_name and the columns root_depth_max_m	Kc_ini	Kc_mid	Kc_end	p
# empty and filled by the user
csv_file = f"{OUTPUT_PATH}/CROPG_fractional.csv"
df = pd.DataFrame({
    "crop_name": crop_names,
    "root_depth_max_m": np.nan,
    "Kc_ini": np.nan,
    "Kc_mid": np.nan,
    "Kc_end": np.nan,
    "p": np.nan
})
df.to_csv(csv_file, index=False)


print(f"Finished processing all crop layers. Output file: {output_file}, CSV file: {csv_file}")