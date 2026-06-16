# FAO-56 crop parameters CSV: Table 22 + Table 12

This README documents the fields in `fao56_crop_parameters_table22_kc.csv`.

The CSV is intended for irrigation-water-requirement or crop-water-balance scripts. It combines:

- FAO-56 Table 22 parameters: rooting depth (`Zr`) and the soil-water depletion fraction for no stress (`p`).
- FAO-56 Table 12 parameters: crop coefficients (`Kc`) and maximum crop height (`h`).

The values are intended as reference/default values for non-stressed, well-managed crops. They should be adjusted or calibrated when local crop varieties, management practices, soil limitations, salinity, climate, or crop calendars differ from FAO-56 assumptions.

---

## How to use the file

Each row represents one crop or crop group. The recommended unique key is:

```text
crop_id
```

Typical use in a model:

1. Match raster band names or crop-map classes to `crop_id`.
2. Use `root_depth_min_m` and `root_depth_max_m` to define the effective rooting depth.
3. Compute total available water:

```text
TAW = 1000 * (theta_FC - theta_WP) * Zr
```

where:

- `TAW` is total available water in the root zone, in mm.
- `theta_FC` is soil water content at field capacity, in m3/m3.
- `theta_WP` is soil water content at wilting point, in m3/m3.
- `Zr` is rooting depth, in m.
- `1000` converts metres of water to mm.

4. Compute readily available water:

```text
RAW = p * TAW
```

5. Use `kc_ini`, `kc_mid`, and `kc_end` to construct the crop coefficient curve and compute crop evapotranspiration:

```text
ETc = Kc * ETo
```

where `ETo` is reference evapotranspiration.

---

## Columns

### Crop identifiers

#### `category`
General FAO-56 crop category, for example `Small vegetables`, `Cereals`, `Legumes`, `Fibres`, `Oil crops`, `Forages`, `Tropical fruits`, or similar.

Useful for grouping crops, filtering tables, or applying fallback parameters when an exact crop match is not available.

#### `crop_id`
Machine-readable crop identifier.

This is the recommended key for code, raster band names, lookup tables, and joins. It is lowercase and uses underscores instead of spaces.

Examples:

```text
maize_grain
soybean
rice
wheat_spring
broccoli
```

#### `crop_name_fao56`
Human-readable crop name as represented in the FAO-56 source table or derived from it.

This field is mainly for readability, reporting, and manual checks.

---

### Rooting depth and soil-water depletion parameters

#### `root_depth_min_m`
Minimum typical effective rooting depth, in metres.

This comes from FAO-56 Table 22. It represents the lower bound of the typical maximum effective rooting depth range for the crop.

#### `root_depth_max_m`
Maximum typical effective rooting depth, in metres.

This comes from FAO-56 Table 22. It represents the upper bound of the typical maximum effective rooting depth range for the crop.

In operational modelling, common choices are:

- use `root_depth_max_m` for mature crops under unrestricted rooting conditions;
- use the mean of min and max where local information is limited;
- use a time-varying rooting depth curve if crop development stages are explicitly simulated;
- cap rooting depth by soil depth or restrictive soil layers where available.

#### `p_table22_for_ETc_5mm_day`
FAO-56 soil-water depletion fraction for no stress, dimensionless.

This is the fraction of total available water (`TAW`) that can be depleted from the root zone before crop water stress starts. It is generally referred to as `p` in FAO-56.

FAO-56 Table 22 values are given for approximately:

```text
ETc = 5 mm/day
```

The parameter is used to compute readily available water:

```text
RAW = p * TAW
```

A lower `p` means the crop is more sensitive to water depletion and stress starts earlier. A higher `p` means the crop can tolerate a larger depletion of root-zone water before stress starts.

#### `p_adjustment_formula`
Text description of the FAO-56 adjustment of `p` for evapotranspiration rates different from 5 mm/day.

The formula included in the CSV is:

```text
p_adj = min(max(p_table22 + 0.04 * (5 - ETc_mm_day), 0.1), 0.8)
```

where:

- `p_table22` is `p_table22_for_ETc_5mm_day`;
- `ETc_mm_day` is crop evapotranspiration in mm/day;
- `p_adj` is the adjusted depletion fraction;
- the result is clipped to a practical range from 0.1 to 0.8.

Interpretation:

- when `ETc` is higher than 5 mm/day, `p` decreases and stress starts earlier;
- when `ETc` is lower than 5 mm/day, `p` increases and stress starts later.

---

### Crop coefficient parameters

The `Kc` values are from FAO-56 Table 12. They are single crop coefficients for non-stressed, well-managed crops under the FAO-56 reference conditions.

The single crop coefficient combines crop transpiration and average soil evaporation effects into one coefficient.

#### `kc_ini`
Initial-stage crop coefficient, dimensionless.

This corresponds to the crop coefficient during the initial growth stage. It is especially sensitive to wetting frequency, soil evaporation, irrigation method, and rainfall frequency. FAO-56 recommends refining `Kc_ini` when local information is available.

Use with care for daily modelling, especially in dry climates or under drip irrigation.

#### `kc_ini_min`
Numeric minimum value parsed from `kc_ini`.

For most crops this is equal to `kc_ini`. If the original value is a range or contains alternatives, this field stores the lower numeric value.

#### `kc_ini_max`
Numeric maximum value parsed from `kc_ini`.

For most crops this is equal to `kc_ini`. If the original value is a range or contains alternatives, this field stores the upper numeric value.

#### `kc_mid`
Mid-season crop coefficient, dimensionless.

This corresponds to the crop coefficient during the mid-season stage, usually when the crop has maximum or near-maximum ground cover and evapotranspiration is near its seasonal maximum.

FAO-56 Table 12 values are for subhumid climates with approximately:

```text
RHmin ≈ 45%
u2 ≈ 2 m/s
```

where `RHmin` is minimum relative humidity and `u2` is wind speed at 2 m height.

For different climatic conditions, `kc_mid` should be adjusted using the FAO-56 climatic correction.

#### `kc_mid_min`
Numeric minimum value parsed from `kc_mid`.

For most crops this is equal to `kc_mid`. If the original value is a range or contains alternatives, this field stores the lower numeric value.

#### `kc_mid_max`
Numeric maximum value parsed from `kc_mid`.

For most crops this is equal to `kc_mid`. If the original value is a range or contains alternatives, this field stores the upper numeric value.

#### `kc_end`
End-season crop coefficient, dimensionless.

This corresponds to the crop coefficient at the end of the late-season stage.

Some crops have alternative end-season values depending on harvest conditions. For example, a crop harvested dry may have a lower `kc_end` than the same crop harvested fresh or before full senescence.

#### `kc_end_min`
Numeric minimum value parsed from `kc_end`.

For most crops this is equal to `kc_end`. If the original value is a range or contains alternatives, this field stores the lower numeric value.

#### `kc_end_max`
Numeric maximum value parsed from `kc_end`.

For most crops this is equal to `kc_end`. If the original value is a range or contains alternatives, this field stores the upper numeric value.

---

### Crop height parameters

#### `max_crop_height_m`
Maximum crop height, in metres, from FAO-56 Table 12.

This value is used in FAO-56 climatic adjustments of `Kc_mid` and `Kc_end`, and can also be useful for aerodynamic or crop-development assumptions.

#### `max_crop_height_min_m`
Numeric minimum value parsed from `max_crop_height_m`.

For most crops this is equal to `max_crop_height_m`. If the original value is a range or contains alternatives, this field stores the lower numeric value.

#### `max_crop_height_max_m`
Numeric maximum value parsed from `max_crop_height_m`.

For most crops this is equal to `max_crop_height_m`. If the original value is a range or contains alternatives, this field stores the upper numeric value.

---

### Metadata and notes

#### `kc_condition`
Short description of how the Kc value was assigned.

Typical values may indicate whether the row comes directly from a standard FAO-56 Table 12 crop row, a related crop row, or another mapped/approximated condition.

Use this column to identify rows that may need manual review.

#### `source`
Source for the rooting depth and depletion fraction values.

Usually:

```text
FAO Irrigation and Drainage Paper 56, Table 22, Chapter 8
```

#### `kc_source`
Source for the crop coefficient and crop height values.

Usually:

```text
FAO Irrigation and Drainage Paper 56, Table 12, Chapter 6
```

#### `notes`
Optional notes related to Table 22 parameters, crop mapping, assumptions, or interpretation.

May be empty.

#### `kc_notes`
Optional notes related to Kc values, alternatives, harvest conditions, or mapping decisions.

May be empty.

---

## Suggested interpretation in code

### Choosing rooting depth

If no dynamic rooting-depth module is used, a simple option is:

```python
zr = root_depth_max_m
```

A more conservative option is:

```python
zr = 0.5 * (root_depth_min_m + root_depth_max_m)
```

A better crop-growth implementation would vary rooting depth over the crop season, increasing from a shallow initial value to the selected maximum rooting depth.

### Choosing Kc values when alternatives are present

If `kc_ini`, `kc_mid`, or `kc_end` contain a single value, use it directly.

If alternatives or ranges are present:

- use the `_min` and `_max` columns for uncertainty bounds;
- choose the most appropriate value based on crop condition, harvest moisture, or local agronomic information;
- keep the original text column for traceability.

### Adjusting p for ETc

Use the Table 22 value directly when no adjustment is needed:

```python
p = p_table22_for_ETc_5mm_day
```

Use the FAO-56 adjustment when daily or period-specific ETc is available:

```python
p_adj = p_table22_for_ETc_5mm_day + 0.04 * (5 - ETc_mm_day)
p_adj = min(max(p_adj, 0.1), 0.8)
```

### Computing water stress threshold

```python
TAW = 1000 * (theta_fc - theta_wp) * zr
RAW = p_adj * TAW
```

Water stress starts when root-zone depletion exceeds `RAW`.

---

## Important limitations

1. The table gives generic FAO-56 reference parameters, not locally calibrated values.
2. Kc values are for standard, non-stressed conditions.
3. Kc values should be adjusted for local climate, especially wind speed and humidity, where needed.
4. Kc_ini is highly sensitive to wetting events and soil evaporation.
5. Rooting depth should be limited by actual soil depth, restrictive layers, groundwater, salinity, and crop development stage.
6. The depletion fraction `p` is crop-specific but also depends on atmospheric demand through ETc.
7. The CSV does not include crop calendars or growth-stage lengths. These should be stored separately because they are location- and season-dependent.
8. Some crop-name matches between Table 22 and Table 12 may require interpretation; review `notes`, `kc_condition`, and `kc_notes` before operational use.

---

## Recommended companion tables

For a complete irrigation requirement workflow, keep this CSV separate from more dynamic or location-specific files, for example:

```text
crop_parameters_fao56.csv       # this file: Zr, p, Kc, height
crop_calendar.csv               # planting date, harvest date, growth stages
soil_parameters.csv or rasters   # theta_FC, theta_WP, soil depth
climate_data                    # ETo, precipitation, temperature, etc.
irrigation_efficiency.csv        # application/conveyance efficiency if gross irrigation is needed
```

---

## Suggested citation

FAO Irrigation and Drainage Paper No. 56: Crop Evapotranspiration - Guidelines for computing crop water requirements. Allen, R.G., Pereira, L.S., Raes, D., and Smith, M. FAO, Rome.
