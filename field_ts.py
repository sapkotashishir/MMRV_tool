"""Core functions: Sentinel-2 NDVI time series for one field from a GEE asset."""
import json
import ee
import numpy as np
import pandas as pd

CS_THRESH = 0.60       # Cloud Score+ threshold
EDGE_BUFFER_M = -10    # drop mixed edge pixels
MIN_VALID_FRAC = 0.5   # min clear fraction of core area per image


def init_ee(sa_info=None, project=None):
    """sa_info = service-account JSON as dict (deployed); None = your own login (local/Colab)."""
    if sa_info:
        creds = ee.ServiceAccountCredentials(
            sa_info["client_email"], key_data=json.dumps(sa_info))
        ee.Initialize(creds, project=project or sa_info.get("project_id"))
    else:
        ee.Initialize(project=project)


def list_field_ids(asset, id_prop="Name"):
    fc = ee.FeatureCollection(asset)

    values = (
        fc.filter(ee.Filter.notNull([id_prop]))
        .aggregate_array(id_prop)
        .distinct()
        .getInfo()
    )

    return sorted(values, key=str)


def field_feature(asset, field_id, id_prop="Name"):
    matches = ee.FeatureCollection(asset).filter(
        ee.Filter.eq(id_prop, field_id)
    )

    count = matches.size().getInfo()

    if count == 0:
        raise ValueError(
            f"No feature found with {id_prop}={field_id!r}"
        )

    if count > 1:
        raise ValueError(
            f"{count} features share {id_prop}={field_id!r}. "
            "Use a unique identifier or explicitly combine them."
        )

    return ee.Feature(matches.first())


def boundary_qc(geom):
    area_ha = geom.area(1).divide(1e4).getInfo()
    core_ha = geom.buffer(EDGE_BUFFER_M).area(1).divide(1e4).getInfo()
    flags = []
    if area_ha < 0.5: flags.append("very small (<0.5 ha)")
    if area_ha > 400: flags.append("very large (>400 ha): check boundary")
    if core_ha < 0.3 * area_ha: flags.append("narrow shape: little clean core at 10 m")
    return dict(area_ha=round(area_ha, 2), core_ha=round(core_ha, 2), flags=flags)


def _s2(core, start, end):
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(core).filterDate(start, end))
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = s2.linkCollection(cs, ["cs_cdf"])

    def prep(img):
        img = img.updateMask(img.select("cs_cdf").gte(CS_THRESH))
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return ndvi.copyProperties(img, ["system:time_start"])
    return s2.map(prep)


def extract_series(geom, start, end):
    # Validate the selected boundary.
    geometry_type = geom.type().getInfo()

    if geometry_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            "The selected feature must be a Polygon or MultiPolygon. "
            f"Found: {geometry_type}"
        )

    core = geom.buffer(EDGE_BUFFER_M, 1)

    if core.area(1).getInfo() <= 0:
        raise ValueError(
            "No usable field area remains after the inward buffer."
        )

    ic = _s2(core, start, end)

    def per_img(img):
        img = ee.Image(img).select("NDVI")
        projection = img.projection()

        # Use exactly the same grid for all three reductions.
        reduction_args = {
            "geometry": core,
            "crs": projection,
            "scale" : 10,
            "maxPixels": 1e9,
            "tileScale": 2,
        }

        mean = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            **reduction_args,
        ).get("NDVI")

        valid_count = ee.Number(
            img.reduceRegion(
                reducer=ee.Reducer.count(),
                **reduction_args,
            ).get("NDVI", 0)
        )

        total_count = ee.Number(
            ee.Image.constant(1)
            .rename("total")
            .reduceRegion(
                reducer=ee.Reducer.count(),
                **reduction_args,
            )
            .get("total", 0)
        )

        valid_frac = ee.Algorithms.If(
            total_count.gt(0),
            valid_count.divide(total_count.max(1)),
            0,
        )

        return ee.Feature(None, {
            "date": ee.Date(
                img.get("system:time_start")
            ).format("YYYY-MM-dd"),
            "NDVI": mean,
            "valid_frac": valid_frac,
        })

    fc = (
        ee.FeatureCollection(ic.map(per_img))
        .filter(ee.Filter.notNull(["NDVI"]))
        .filter(
            ee.Filter.gte("valid_frac", MIN_VALID_FRAC)
        )
    )

    rows = [
        feature["properties"]
        for feature in fc.getInfo()["features"]
    ]

    columns = [
        "date",
        "NDVI",
        "mean_valid_frac",
        "image_count",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            NDVI=("NDVI", "mean"),
            mean_valid_frac=("valid_frac", "mean"),
            image_count=("NDVI", "size"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily


def spike_clean(vals, window=5, sigma=2.5):
    vals = np.asarray(vals, float)
    out, flag, half = vals.copy(), np.zeros(len(vals), bool), window // 2
    for i in range(len(vals)):
        nb = np.r_[vals[max(0, i - half):i], vals[i + 1:i + half + 1]]
        nb = nb[~np.isnan(nb)]
        if len(nb) >= 2 and nb.std() > 0 and abs(vals[i] - nb.mean()) > sigma * nb.std():
            out[i], flag[i] = nb.mean(), True
    return out, flag


def clean_series(df, var="NDVI"):
    df = df.copy()
    df[var + "_clean"], df["spike"] = spike_clean(df[var])
    return df

def s2_tile_urls(asset, field_id, date_str, id_prop="Name"):
    """Tile URLs (for folium) of the Sentinel-2 image(s) on one date."""
    geom = field_feature(asset, field_id, id_prop).geometry()
    d0 = ee.Date(date_str)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(geom).filterDate(d0, d0.advance(1, "day")))
    if col.size().getInfo() == 0:
        return {}

    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    col = col.linkCollection(cs, ["cs_cdf"])

    raw = col.mosaic()                                   # unmasked: you see clouds
    masked = col.map(
        lambda i: i.updateMask(i.select("cs_cdf").gte(CS_THRESH))).mosaic()
    ndvi = masked.normalizedDifference(["B8", "B4"]).rename("NDVI")

    layers = {
        "True color": (raw, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}),
        "False color (NIR)": (raw, {"bands": ["B8", "B4", "B3"], "min": 0, "max": 4000}),
        "NDVI (cloud-masked)": (ndvi, {
            "min": 0, "max": 0.9,
            "palette": ["#8c510a", "#d8b365", "#f6e8c3", "#c7e9b4", "#41ab5d", "#006d2c"]}),
    }
    return {name: img.clip(geom).getMapId(vis)["tile_fetcher"].url_format
            for name, (img, vis) in layers.items()}