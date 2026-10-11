# Original data: where it comes from

**File:** `ph_heat_index_next_day_16cities.csv` (64,240 rows × 26 columns: 16 NCR cities × 4,015 days, 2015-01-01 to 2025-12-28)
SHA-256: `b954754a5d5fecb5e8665d8efdfaeae6ba60376bfe3ce9c986ce5f9516332ed3`
Column descriptions: `ph_heat_index_data_dictionary.csv` (same folder).

## How it was collected

* **Weather:** hourly 2 m temperature, relative humidity, dew point, pressure, cloud cover, rain, shortwave radiation and wind from the
  [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), model `era5` (ERA5 reanalysis,
  about 25 km grid), time zone Asia/Manila, 2015-01-01 to 2025-12-29. Downloaded on 2026-10-07, one request per city and year.
* **Heat index:** computed for every hour from temperature and humidity with the NOAA/NWS (Rothfusz) heat index equation; the
  highest hour of the day is `HeatIndex_Max_Today`; `HeatIndex_Max_Tomorrow` is the next day's value.
* **Cities:** the 16 cities of the National Capital Region from the Philippine Statistics Authority's PSGC list (via the `psgc` package, v2026.4.13.0).
  Each city's coordinates are the population-weighted mean of its barangay centres (OCHA/HDX boundaries).
* **Daily values:** hourly values summarised to one value per city per day (minimum, mean, maximum or total).

The code that did this, the raw API responses and a longer analysis are in the earlier repository
[`keyyum/heatcast-ph`](https://github.com/keyyum/heatcast-ph) (`build_ph_heat_index_dataset.py`, `data/ncr/cache/raw/`, `data/ncr/README.md`).

## Licence and ethics

Public, non-personal weather and administrative data. Open-Meteo: CC BY 4.0, free API for **non-commercial** use (this is a course project).
Contains modified Copernicus Climate Change Service (ERA5) information. OCHA/HDX boundaries: CC BY-IGO. `psgc` package: MIT; PSGC data are public information of the Philippine Statistics Authority.

## Caution

The 16 cities fall in only 2 ERA5 grid cells, so they are **not 16 independent weather stations**. That is why the simple dataset averages them into one row per day. The average hides small differences between cities (on the same day the hottest and coolest city differ by a median of 1.75 °C in the heat index).
The data are weather-model estimates, not station observations.
