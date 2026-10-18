"""Additional datasets for the field-monitoring app."""

import ee
import pandas as pd

import field_ts as ft


SCALE = {
    "Sentinel-1": 10,
    "Landsat": 30,
    "HLS": 30,
}

NDVI_VIS = {
    "bands": ["NDVI"],
    "min": 0,
    "max": 0.9,
    "palette": [
        "#8c510a",
        "#d8b365",
        "#f6e8c3",
        "#c7e9b4",
        "#41ab5d",
        "#006d2c",
    ],
}


def _s1_base(geom, start, end, orbit_pass):
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("resolution_meters", 10))
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            )
        )
        .filter(
            ee.Filter.eq("orbitProperties_pass", orbit_pass)
        )
    )


def s1_orbits(geom, start, end, orbit_pass):
    values = (
        _s1_base(geom, start, end, orbit_pass)
        .aggregate_array("relativeOrbitNumber_start")
        .distinct()
        .getInfo()
    )
    return sorted(int(value) for value in values)


def _add_ndvi(image):
    nir = image.select("nir")
    red = image.select("red")
    denominator = nir.add(red)

    ndvi = (
        nir.subtract(red)
        .divide(denominator)
        .updateMask(denominator.neq(0))
        .rename("NDVI")
    )

    return image.addBands(ndvi)


def _prep_landsat(image):
    # Reject fill, dilated cloud, cirrus, cloud, shadow, snow.
    qa_ok = image.select("QA_PIXEL").bitwiseAnd(63).eq(0)
    saturation_ok = image.select("QA_RADSAT").eq(0)

    reflectance = (
        image.select(
            ["SR_B2", "SR_B3", "SR_B4", "SR_B5"],
            ["blue", "green", "red", "nir"],
        )
        .multiply(0.0000275)
        .add(-0.2)
        .updateMask(qa_ok.And(saturation_ok))
    )

    return _add_ndvi(reflectance).copyProperties(
        image, ["system:time_start", "system:index"]
    )


def _prep_hls(image, nir_band):
    qa = image.select("Fmask")

    # Reject cloud, adjacent cloud/shadow, shadow, snow.
    clear = qa.bitwiseAnd(30).eq(0)

    # Also reject the high-aerosol category.
    aerosol_ok = qa.rightShift(6).bitwiseAnd(3).lt(3)

    reflectance = (
        image.select(
            ["B2", "B3", "B4", nir_band],
            ["blue", "green", "red", "nir"],
        )
        .updateMask(clear.And(aerosol_ok))
    )

    return _add_ndvi(reflectance).copyProperties(
        image, ["system:time_start", "system:index"]
    )


def collection(
    source,
    geom,
    start,
    end,
    orbit_pass="ASCENDING",
    relative_orbit=None,
):
    if source == "Sentinel-1":
        if relative_orbit is None:
            raise ValueError("Select a Sentinel-1 relative orbit.")

        return (
            _s1_base(geom, start, end, orbit_pass)
            .filter(
                ee.Filter.eq(
                    "relativeOrbitNumber_start",
                    int(relative_orbit),
                )
            )
            .select(["VV", "VH"])
            .sort("system:time_start")
        )

    if source == "Landsat":
        l8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(geom)
            .filterDate(start, end)
            .map(_prep_landsat)
        )

        l9 = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(geom)
            .filterDate(start, end)
            .map(_prep_landsat)
        )

        return l8.merge(l9).sort("system:time_start")

    if source == "HLS":
        l30 = (
            ee.ImageCollection("NASA/HLS/HLSL30/v002")
            .filterBounds(geom)
            .filterDate(start, end)
            .map(lambda image: _prep_hls(image, "B5"))
        )

        s30 = (
            ee.ImageCollection("NASA/HLS/HLSS30/v002")
            .filterBounds(geom)
            .filterDate(start, end)
            .map(lambda image: _prep_hls(image, "B8A"))
        )

        return l30.merge(s30).sort("system:time_start")

    raise ValueError(f"Unsupported dataset: {source}")


def extract_series(
    geom,
    start,
    end,
    source,
    variable,
    orbit_pass="ASCENDING",
    relative_orbit=None,
):
    if geom.type().getInfo() not in ("Polygon", "MultiPolygon"):
        raise ValueError("Select a polygon field boundary.")

    core = geom.buffer(ft.EDGE_BUFFER_M, 1)

    if core.area(1).getInfo() <= 0:
        raise ValueError(
            "No usable area remains after the inward buffer."
        )

    ic = collection(
        source, geom, start, end, orbit_pass, relative_orbit
    )

    def per_image(image):
        band = ee.Image(image).select(variable)

        args = {
            "geometry": core,
            "crs": band.projection(),
            "scale": SCALE[source],
            "maxPixels": 1e9,
            "tileScale": 2,
        }

        mean = band.reduceRegion(
            reducer=ee.Reducer.mean(), **args
        ).get(variable)

        valid = ee.Number(
            band.reduceRegion(
                reducer=ee.Reducer.count(), **args
            ).get(variable, 0)
        )

        total = ee.Number(
            ee.Image.constant(1)
            .rename("total")
            .reduceRegion(
                reducer=ee.Reducer.count(), **args
            )
            .get("total", 0)
        )

        fraction = ee.Algorithms.If(
            total.gt(0),
            valid.divide(total.max(1)),
            0,
        )

        return ee.Feature(None, {
            "date": ee.Date(
                image.get("system:time_start")
            ).format("YYYY-MM-dd"),
            variable: mean,
            "valid_frac": fraction,
        })

    table = (
        ee.FeatureCollection(ic.map(per_image))
        .filter(ee.Filter.notNull([variable]))
        .filter(
            ee.Filter.gte("valid_frac", ft.MIN_VALID_FRAC)
        )
    )

    rows = [
        feature["properties"]
        for feature in table.getInfo()["features"]
    ]

    columns = [
        "date",
        variable,
        "mean_valid_frac",
        "image_count",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    daily = df.groupby("date", as_index=False).agg(
        **{
            variable: (variable, "mean"),
            "mean_valid_frac": ("valid_frac", "mean"),
            "image_count": (variable, "size"),
        }
    )

    daily["dataset"] = source

    if source == "Sentinel-1":
        daily["orbit_pass"] = orbit_pass
        daily["relative_orbit"] = relative_orbit

    return daily.sort_values("date").reset_index(drop=True)


def tile_urls(
    asset,
    field_id,
    date_str,
    source,
    orbit_pass="ASCENDING",
    relative_orbit=None,
):
    geom = ft.field_feature(asset, field_id).geometry()
    day = ee.Date(date_str)

    ic = collection(
        source,
        geom,
        day,
        day.advance(1, "day"),
        orbit_pass,
        relative_orbit,
    )

    if ic.size().getInfo() == 0:
        return {}

    image = ic.mosaic()

    if source == "Sentinel-1":
        layers = {
            "VV (dB)": (
                image.select("VV"),
                {
                    "min": -25,
                    "max": 0,
                    "palette": ["black", "white"],
                },
            ),
            "VH (dB)": (
                image.select("VH"),
                {
                    "min": -30,
                    "max": -5,
                    "palette": ["black", "white"],
                },
            ),
        }
    else:
        layers = {
            "True color": (
                image,
                {
                    "bands": ["red", "green", "blue"],
                    "min": 0,
                    "max": 0.3,
                },
            ),
            "False color (NIR)": (
                image,
                {
                    "bands": ["nir", "red", "green"],
                    "min": 0,
                    "max": 0.4,
                },
            ),
            "NDVI (cloud-masked)": (
                image.select("NDVI"),
                NDVI_VIS,
            ),
        }

    return {
        name: layer.clip(geom)
        .getMapId(vis)["tile_fetcher"]
        .url_format
        for name, (layer, vis) in layers.items()
    }