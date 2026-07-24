#!/usr/bin/env python3

import argparse
import calendar
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio


NODATA = 65535

REQUIRED_KEYS = (
    "phenos1",
    "phenom1",
    "phenosen1",
    "phenoe1",
    "phenos2",
    "phenom2",
    "phenosen2",
    "phenoe2",
    "phenonseasons",
)


def read_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    paths = config["phenology_paths"]

    missing = [key for key in REQUIRED_KEYS if key not in paths]
    if missing:
        raise KeyError(f"Missing phenology paths: {', '.join(missing)}")

    return config, {key: Path(paths[key]) for key in REQUIRED_KEYS}


def read_layers(paths):
    reference_path = paths["phenonseasons"]

    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs

    layers = {}
    invalid = {}

    for name, path in paths.items():
        with rasterio.open(path) as src:
            if (
                src.width != width
                or src.height != height
                or src.crs != crs
                or not src.transform.almost_equals(transform)
            ):
                raise ValueError(f"Raster not aligned: {path}")

            array = src.read(1, masked=True).astype(np.float32)
            layers[name] = array.filled(np.nan)
            invalid[name] = np.ma.getmaskarray(array) | ~np.isfinite(array)

    return layers, invalid, profile


def dekad_lengths(year):
    lengths = []

    for month in range(1, 13):
        days = calendar.monthrange(year, month)[1]
        lengths.extend((10, 10, days - 20))

    return lengths


def season_status(current_dekad, sos, tom, sen, eos, valid):
    status = np.zeros(sos.shape, dtype=np.uint8)

    for value in (
        current_dekad,
        current_dekad + 36,
        current_dekad + 72,
    ):
        active = valid & (value >= sos) & (value <= eos)

        status[
            active
            & (value >= sos)
            & (value < tom)
        ] = 1

        status[
            active
            & (value >= tom)
            & (value < sen)
        ] = 2

        status[
            active
            & (value >= sen)
            & (value <= eos)
        ] = 3

    return status


def count_season_days(sos, tom, sen, eos, valid, year):
    growing = np.zeros(sos.shape, dtype=np.uint16)
    maximum = np.zeros(sos.shape, dtype=np.uint16)
    senescence = np.zeros(sos.shape, dtype=np.uint16)

    for dekad, days in enumerate(dekad_lengths(year), start=1):
        status = season_status(
            current_dekad=dekad,
            sos=sos,
            tom=tom,
            sen=sen,
            eos=eos,
            valid=valid,
        )

        growing[status == 1] += days
        maximum[status == 2] += days
        senescence[status == 3] += days

    return growing, maximum, senescence


def create_output_arrays(layers, invalid, year):
    seasons = layers["phenonseasons"]

    output_invalid = invalid["phenonseasons"].copy()
    output_invalid |= (seasons < 0) | (seasons > 2)

    growing_total = np.zeros(seasons.shape, dtype=np.uint16)
    maximum_total = np.zeros(seasons.shape, dtype=np.uint16)
    senescence_total = np.zeros(seasons.shape, dtype=np.uint16)

    for season_number in (1, 2):
        suffix = str(season_number)

        sos_key = f"phenos{suffix}"
        tom_key = f"phenom{suffix}"
        sen_key = f"phenosen{suffix}"
        eos_key = f"phenoe{suffix}"

        requested = (
            ~invalid["phenonseasons"]
            & (seasons >= season_number)
        )

        valid = (
            requested
            & ~invalid[sos_key]
            & ~invalid[tom_key]
            & ~invalid[sen_key]
            & ~invalid[eos_key]
        )

        sos = layers[sos_key]
        tom = layers[tom_key]
        sen = layers[sen_key]
        eos = layers[eos_key]

        valid &= (
            (sos >= 1)
            & (eos <= 108)
            & (sos <= tom)
            & (tom <= sen)
            & (sen <= eos)
        )

        output_invalid |= requested & ~valid

        growing, maximum, senescence = count_season_days(
            sos=sos,
            tom=tom,
            sen=sen,
            eos=eos,
            valid=valid,
            year=year,
        )

        growing_total += growing
        maximum_total += maximum
        senescence_total += senescence

    growing_total[output_invalid] = NODATA
    maximum_total[output_invalid] = NODATA
    senescence_total[output_invalid] = NODATA
    active_total = np.where(
        output_invalid,
        NODATA,
        growing_total + maximum_total + senescence_total,
    ).astype(np.uint16)

    return growing_total, maximum_total, senescence_total, active_total


def write_geotiff(path, profile, growing, maximum, senescence, active, year):
    path.parent.mkdir(parents=True, exist_ok=True)

    profile.update(
        driver="GTiff",
        count=4,
        dtype="uint16",
        nodata=NODATA,
        compress="deflate",
        predictor=2,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(growing, 1)
        dst.write(maximum, 2)
        dst.write(senescence, 3)
        dst.write(active, 4)

        dst.set_band_description(1, "growing_days")
        dst.set_band_description(2, "maximum_days")
        dst.set_band_description(3, "senescence_days")
        dst.set_band_description(4, "active_days_total")

        dst.update_tags(1, units="days")
        dst.update_tags(2, units="days")
        dst.update_tags(3, units="days")
        dst.update_tags(4, units="days")
        dst.update_tags(reference_year=str(year))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    config, paths = read_config(args.config)

    if args.reference_year is not None:
        year = args.reference_year
    else:
        year = datetime.strptime(
            config["start_date"],
            "%Y-%m-%d",
        ).year

    output = args.output
    if output is None:
        output = (
            paths["phenonseasons"].parent
            / "phenology_phase_duration_days.tif"
        )

    layers, invalid, profile = read_layers(paths)

    growing, maximum, senescence, active = create_output_arrays(
        layers=layers,
        invalid=invalid,
        year=year,
    )

    write_geotiff(
        path=output,
        profile=profile,
        growing=growing,
        maximum=maximum,
        senescence=senescence,
        active=active,
        year=year,
    )

    print(output)


if __name__ == "__main__":
    main()
