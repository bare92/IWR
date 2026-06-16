# Pure WATNEEDS Green Water Step Function

**Location**: [src/iwr_processing/iwr_core_process.py](src/iwr_processing/iwr_core_process.py#L103)

## Pure Function: `watneeds_green_step()`

Pure, vectorized implementation of the WATNEEDS daily water balance, independent of class methods or state.

### Signature

```python
def watneeds_green_step(
    s_prev_mm: np.ndarray,
    precipitation_mm: np.ndarray,
    etc_mm: np.ndarray,
    taw_mm: np.ndarray,
    p: np.ndarray | float,
    fmax_mm_day: np.ndarray,
    peff_coeff: float = 0.95,
    dt_days: float = 1.0,
    irrigated_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
```

### Parameters

| Param | Type | Shape | Description |
|-------|------|-------|-------------|
| `s_prev_mm` | ndarray | (rows, cols) | Soil moisture at day start (mm) in [0, TAW] |
| `precipitation_mm` | ndarray | (rows, cols) | Daily precipitation (mm) |
| `etc_mm` | ndarray | (rows, cols) or scalar | Crop ET (mm/day) |
| `taw_mm` | ndarray | (rows, cols) | Total available water (mm) |
| `p` | float or ndarray | scalar or (rows, cols) | Depletion fraction for stress threshold [0.3, 0.6] |
| `fmax_mm_day` | ndarray | (rows, cols) | Max deep percolation (mm/day) |
| `peff_coeff` | float | scalar | Effective precip coeff, default 0.95 (5% runoff) |
| `dt_days` | float | scalar | Timestep in days, default 1.0 |
| `irrigated_mask` | ndarray or None | (rows, cols) bool | True=irrigated, False=rainfed. If None, no blue water. |
| `valid_mask` | ndarray or None | (rows, cols) bool | True=valid pixel. Outputs zeroed outside. |

### Returns

Dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `s_next_mm` | ndarray | Soil moisture at day end, clipped [0, TAW] |
| `green_et_mm` | ndarray | Green ET = actual ET under stress (mm) |
| `deep_perc_mm` | ndarray | Deep percolation loss (mm) |
| `blue_water_mm` | ndarray | Blue water requirement = max(ETc - GW, 0) (mm) |
| `ks` | ndarray | Water stress coefficient [0, 1] |
| `surface_runoff_mm` | ndarray | Runoff from precipitation: (1 - peff_coeff) * P |
| `overflow_runoff_mm` | ndarray | Runoff from saturation: max(S_after - TAW, 0) |
| `total_runoff_mm` | ndarray | surface_runoff + overflow_runoff |
| `peff_mm` | ndarray | Effective precipitation: peff_coeff * P |
| `available_mm` | ndarray | Total water available: S_prev + Peff |

## Algorithm

**1. Effective Precipitation**
```
Peff = peff_coeff * P * dt
surface_runoff = (1 - peff_coeff) * P * dt
```

**2. Stress Threshold (RAW)**
```
threshold = (1 - p) * TAW
```

**3. Water Stress Coefficient (linear)**
```
Ks = 1.0 if S_prev >= threshold
Ks = S_prev / threshold if S_prev < threshold
Ks ∈ [0, 1]
```

**4. Green ET**
```
Green_ET = Ks * ETc
```

**5. Deep Percolation (linear from threshold to TAW)**
```
DP = Fmax * (S_prev - threshold) / (TAW - threshold)  if S_prev >= threshold
DP = 0  if S_prev < threshold
DP ∈ [0, Fmax * dt]
```

**6. Negative Balance Scaling**
```
available = S_prev + Peff
losses = Green_ET + DP
if available < losses:
  scale = available / losses  ∈ [0, 1]
  Green_ET *= scale
  DP *= scale
```

**7. Soil Moisture Update & Overflow**
```
S_after = available - Green_ET - DP
overflow_runoff = max(S_after - TAW, 0)
S_next = clip(S_after, [0, TAW])
```

**8. Blue Water (on irrigated pixels only)**
```
BW = max(ETc - Green_ET, 0)
BW = 0 if rainfed or irrigated_mask = False
```

## Key Differences from Original IWRModel.watneeds_soil_water_balance_step()

| Aspect | Pure Function | Class Method |
|--------|---------------|-------------|
| **State** | None—pure inputs/outputs | Manages soil, crop, grids internally |
| **Masking** | Optional valid_mask | Automatic hydraulic_mask() |
| **Crop params** | Caller provides p, taw, fmax | Loaded from soil/crop layers |
| **Vectorization** | Full numpy arrays only | Accepts raster paths, scalars |
| **Use case** | High-performance kernel, time-series loops | Convenient single-step API |

## Usage Examples

### Example 1: Direct Vectorized Call

```python
import numpy as np
from iwr_processing.iwr_core_process import watneeds_green_step

# Load your arrays
s_prev = np.load("soil_moisture_mm.npy")  # shape (rows, cols)
precip = np.load("precipitation_mm.npy")
etc = np.load("etc_mm_day.npy")
taw = np.load("taw_mm.npy")
fmax = np.load("fmax_mm_day.npy")
irr_mask = np.load("irrigated.npy").astype(bool)

# Call pure function
result = watneeds_green_step(
    s_prev_mm=s_prev,
    precipitation_mm=precip,
    etc_mm=etc,
    taw_mm=taw,
    p=0.5,  # FAO-56 typical
    fmax_mm_day=fmax,
    peff_coeff=0.95,
    dt_days=1.0,
    irrigated_mask=irr_mask,
)

# Extract outputs
s_next = result["s_next_mm"]
green_et = result["green_et_mm"]
blue_water = result["blue_water_mm"]
```

### Example 2: Through IWRModel.green_water_step()

```python
from iwr_processing.iwr_core_process import IWRModel

model = IWRModel(
    land_cover_path="crops.tif",
    soil_path="soil_class.tif",
    taw_layer="taw_mm.tif",  # WATNEEDS external input
    fmax_layer="fmax_mm_day.tif"  # WATNEEDS external input
)

result = model.green_water_step(
    crop="wheat",
    s_prev_mm="soil_moisture_day1.tif",
    precipitation_mm="precip_day1.tif",
    etc_mm="etc_day1.tif",
    irrigated_mask="irrigated_mask.tif",
    peff_coeff=0.95,
    dt_days=1.0,
)

# Same outputs as pure function
s_next = result["s_next_mm"]
green_et = result["green_et_mm"]
blue_water = result["blue_water_mm"]
```

### Example 3: Daily Loop with Time-Series Driver

```python
from iwr_processing.iwr_core_process import IWRModel, watneeds_green_step
from iwr_processing.crop_calendar import CropCalendar
import numpy as np

model = IWRModel(...)

# Initialize
s_prev = model.get_crop_initial_soil_water_mm("wheat")
daily_results = []

# Daily loop (simplified)
for day in range(365):
    result = watneeds_green_step(
        s_prev_mm=s_prev,
        precipitation_mm=forcing_data["precip"][day],
        etc_mm=forcing_data["etc"][day],
        taw_mm=model.get_crop_taw("wheat"),
        p=model.crop_fractions.get_crop_p("wheat"),
        fmax_mm_day=model._default_fmax_mm_day_from_soil(),
        irrigated_mask=irrigated_mask,
    )
    
    s_prev = result["s_next_mm"]  # Update state for next step
    daily_results.append(result)

# Aggregate
annual_green_water = np.sum([r["green_et_mm"] for r in daily_results], axis=0)
annual_blue_water = np.sum([r["blue_water_mm"] for r in daily_results], axis=0)
```

## Design Rationale

- **Pure function** → No hidden state, trivial to parallelize, composable with time-series drivers
- **Vectorized** → NumPy broadcasting, efficient for large grids
- **Minimal defaults** → Only peff_coeff, dt_days defaulted; soil/crop params caller-provided
- **Explicit masking** → irrigated_mask and valid_mask decoupled from data logic
- **Stress linear** → FAO-56 approach: Ks rises linearly from 0 at WP to 1 at RAW threshold
- **Deep percolation linear** → WATNEEDS approach: DP rises linearly from 0 at threshold to Fmax at TAW
- **Negative balance scaling** → Distributes shortfall proportionally to ET and DP

## References

- **FAO-56**: Allen, L. S., et al. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper 56.
- **WATNEEDS**: Siebert, S., & Döll, P. (2010). Quantifying blue water use for each crop, report 201.
- **MIRCA**: Mueller, N. D., et al. (2017). A global gridded dataset of 5 arcmin monthly precipitation and temperature data.
