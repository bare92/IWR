# Issue H: Crop Calendars and Kc Curves

**Status**: Architecture designed, implementation deferred.

**Problem**: WATNEEDS solves daily balances over full growing seasons with time-varying crop coefficients. Current kernel only handles single-day steps with ETc or ET0/Kc as direct inputs. Missing:
- Crop calendars (planting/harvest dates)
- Time-varying Kc curves (growth stages)
- Full-year simulation with spin-up
- Daily → monthly/yearly aggregation

**Current Scope**: Low-level daily balance kernel only.
- `iwr_core_process.py`: Pure daily water-balance equations
- Input: One-day forcing (precip, ETc or ET0), one-day soil state
- Output: One-day balance terms (ks, eta, D, runoff, etc.)

**To Be Added**: Time-series orchestration layer.
- `crop_calendar.py`: Data structures for calendars and Kc schedules (skeleton created)
- `TimeSeriesDriver`: Daily loop over growing seasons with Kc from calendars
- Aggregation functions: daily → monthly → seasonal → yearly

## Architecture

### Data Sources (TBD)

User is researching options:
- **FAO-56**: Crop coefficients and growth stages from Allen et al. (1998)
- **MIRCA**: Monthly Irrigated and Rainfed Crop Areas (Mueller et al. 2017)
- **Siebert-Döll**: WATNEEDS original datasource
- **Local agronomic data**: Regional extensions or field observations

Once datasource is finalized, implement:
1. Loader for calendars (CSV, NetCDF, raster, or database)
2. Spatial handling (per-pixel calendars vs. uniform regions)
3. Multi-year scenarios

### Data Structures (Created)

**CropGrowthSchedule** (in crop_calendar.py):
- Kc_ini, Kc_mid, Kc_end (FAO-56 nomenclature)
- Kc_off_season (default 0.5)
- Growth-stage durations (initial, development, mid-season, late-season)

**CropCalendar** (in crop_calendar.py):
- Planting DOY, harvest DOY
- Cross-year handling (harvest after Dec 31)
- Query functions: `is_growing_season()`, `days_since_planting()`

**TimeSeriesDriver** (placeholder class in crop_calendar.py):
- Orchestrates daily water balance over full year
- Manages spin-up → analysis → aggregation
- Future: Per-pixel crop calendars, seasonal aggregation

### Integration Points

1. **After loading calendars**: `iwr_model.watneeds_spinup()` (already implemented)
   ```python
   calendar = crop_calendars[(crop_id, year)]
   iwr_model.watneeds_spinup(crop=crop_id, precip_series=..., 
                              et0_kc_series=..., spinup_years=3)
   ```

2. **Daily loop**: `iwr_model.watneeds_soil_water_balance_step()`
   ```python
   kc = compute_kc_daily(calendar, doy, year)
   etc_mm_day = kc * et0_mm_day  # Kc from calendar, ET0 from forcing
   result = iwr_model.watneeds_soil_water_balance_step(
       crop=crop_id, st_prev_mm=S_t, precipitation_mm_day=P_t,
       etc_mm_day=etc_mm_day, ...
   )
   ```

3. **Aggregation**: Sum daily outputs per pixel, then apply crop fractions
   ```python
   blue_water_seasonal = np.sum(daily_blue_water, axis=-1) * crop_fractions[crop]
   ```

## Implementation Roadmap

**Phase 1** (current): Skeleton and documentation
- ✓ CropCalendar and CropGrowthSchedule dataclasses
- ✓ compute_kc_daily() placeholder
- ✓ TimeSeriesDriver class outline
- ✓ Aggregation function stubs

**Phase 2** (when datasource finalized):
- Implement `load_crop_calendars_from_csv()` for your data format
- Implement `compute_kc_daily()` with linear stage interpolation (FAO-56)
- Implement TimeSeriesDriver.run() loop

**Phase 3** (optional):
- Per-pixel calendars (spatial variation)
- Seasonal segmentation (early, mid, late harvest)
- Crop-switching scenarios

## Example Usage (Pseudo-code, not yet runnable)

```python
from iwr_processing.crop_calendar import CropCalendar, CropGrowthSchedule, TimeSeriesDriver
from iwr_processing.iwr_core_process import IWRModel

# 1. Initialize model kernel
model = IWRModel(
    land_cover_path="crops.tif",
    soil_path="soil_class.tif",
    taw_layer="taw_mm.tif",  # WATNEEDS-compliant soil input
    fmax_layer="fmax_mm_day.tif"
)

# 2. Load calendars (implementation TBD)
# calendars = load_crop_calendars_from_csv("calendars.csv")

# 3. Run time-series simulation
# driver = TimeSeriesDriver(
#     iwr_model=model,
#     crop_calendars=calendars,
#     start_date=date(2020, 1, 1),
#     end_date=date(2020, 12, 31),
#     spinup_years=3
# )
# driver.run()

# 4. Get outputs
# daily_blue_water = driver.get_daily_outputs()
# monthly_aggregates = driver.get_monthly_aggregates()
# seasonal_iwr = driver.get_seasonal_aggregates()
```

## Notes

- The daily kernel (`watneeds_soil_water_balance_step`) is complete and tested.
- Crop calendars layer is decoupled from the kernel for modularity.
- Kc computation can be as simple (fixed per stage) or complex (multi-parameter) as datasource allows.
- Off-season Kc = 0.5 is FAO-56 default; adjust in CropGrowthSchedule if needed.
