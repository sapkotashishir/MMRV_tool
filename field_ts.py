"""Core functions: Sentinel-2 time series for one field from a GEE asset."""
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
        ee.Initialize(creds, project=project or sa_info.get("shishirbeeville"))
    else:
        ee.Initialize(project=project)


def list_field_ids(asset, id_prop="Name"):
    return sorted(ee.FeatureCollection(asset).aggregate_array(id_prop).getInfo())


def field_feature(asset, field_id, id_prop="Name"):
    return ee.Feature(ee.FeatureCollection(asset)
                      .filter(ee.Filter.eq(id_prop, field_id)).first())


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
        nbr2 = img.normalizedDifference(["B11", "B12"]).rename("NBR2")
        return ndvi.addBands(nbr2).copyProperties(img, ["system:time_start"])
    return s2.map(prep)


def extract_series(geom, start, end):
    core = geom.buffer(EDGE_BUFFER_M)
    ic = _s2(core, start, end)
    n_total = ee.Image.constant(1).reduceRegion(
        ee.Reducer.count(), core, 10, maxPixels=1e9).get("constant")

    def per_img(img):
        red = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=False)
        st = img.reduceRegion(red, core, 10, maxPixels=1e9)
        return ee.Feature(None, {
            "date": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
            "NDVI": st.get("NDVI_mean"), "NBR2": st.get("NBR2_mean"),
            "valid_frac": ee.Number(st.get("NDVI_count")).divide(n_total)})

    fc = ee.FeatureCollection(ic.map(per_img)).filter(ee.Filter.notNull(["NDVI"]))
    rows = [f["properties"] for f in fc.getInfo()["features"]]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["valid_frac"] >= MIN_VALID_FRAC]
    df = df.groupby("date", as_index=False)[["NDVI", "NBR2"]].mean()  # same-date averaging
    return df.sort_values("date").reset_index(drop=True)


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