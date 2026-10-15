"""Streamlit app: pick a field -> see boundary + Sentinel-2 time series."""
import datetime as dt
import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import field_ts as ft

ASSET = st.secrets.get("asset", "users/Forage_precision/texas_fields")  # your asset
st.set_page_config(page_title="Field Practice Monitor", layout="wide")


@st.cache_resource
def _init():
    sa = dict(st.secrets["gee"]) if "gee" in st.secrets else None
    ft.init_ee(sa, project=st.secrets.get("project"))
    return True

_init()


@st.cache_data(ttl=24 * 3600)
def field_ids():
    return ft.list_field_ids(ASSET)


@st.cache_data(ttl=24 * 3600, show_spinner="Reading Sentinel-2 ...")
def load(field_id, start, end):
    feat = ft.field_feature(ASSET, field_id)
    geom = feat.geometry()
    qc = ft.boundary_qc(geom)
    series = ft.extract_series(geom, start, end)
    return qc, geom.getInfo(), series


st.title("Field Practice Monitor (Sentinel-2)")
with st.sidebar:
    fid = st.selectbox("Field", field_ids())
    start = st.date_input("Start", dt.date(2025, 1, 1))
    end = st.date_input("End", dt.date.today())
    var = st.radio("Index", ["NDVI", "NBR2"])
    reported = st.date_input("Farmer-reported practice date (optional)", value=None)

qc, gj, series = load(fid, str(start), str(end))
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("Boundary check")
    st.write(f"Area: **{qc['area_ha']} ha** (clean core {qc['core_ha']} ha)")
    for f in qc["flags"]:
        st.warning(f)
    ctr = gj["coordinates"][0][0] if gj["type"] == "Polygon" else gj["coordinates"][0][0][0]
    m = folium.Map(location=[ctr[1], ctr[0]], zoom_start=16, tiles=None)
    folium.TileLayer("Esri.WorldImagery", name="Satellite").add_to(m)
    folium.GeoJson(gj, style_function=lambda _: {"color": "red", "fill": False}).add_to(m)
    st_folium(m, height=350, returned_objects=[])

with c2:
    st.subheader(f"{var} time series")
    if series.empty:
        st.info("No clear Sentinel-2 observations in this range.")
    else:
        s = ft.clean_series(series, var)
        long = pd.melt(s, id_vars="date", value_vars=[var, var + "_clean"],
                       var_name="series", value_name="value")
        chart = alt.Chart(long).mark_line(point=True).encode(
            x="date:T", y=alt.Y("value:Q", title=var), color="series:N").properties(height=350)
        if reported:
            rule = alt.Chart(pd.DataFrame({"d": [pd.Timestamp(reported)]})).mark_rule(
                color="orange", strokeDash=[4, 4]).encode(x="d:T")
            chart = chart + rule
        st.altair_chart(chart, use_container_width=True)
        st.download_button("Download CSV", s.to_csv(index=False), f"{fid}_{var}.csv")