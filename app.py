"""Texas field monitoring: Sentinel-2, Sentinel-1, Landsat, and HLS."""

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import altair as alt
import ee
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import field_ts as ft
import satellite_ts as sat


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Texas Field Practice Monitor",
    layout="wide",
)


# --------------------------------------------------
# Application configuration
# --------------------------------------------------
GEE_PROJECT = "shishirbeeville"

# Keep private-key files outside this repository.
# Optionally override this path through an environment variable.
LOCAL_KEY_FILE = Path(
    os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "/home/shishir/Downloads/shishirbeeville-7aa95b746c94.json",
    )
).expanduser()

ASSETS = {
    "Michael Thompson": (
        "users/Forage_precision/Michael-Thompson_1"
    ),
    "Grant Wilde": (
        "users/Forage_precision/Grant-Wilde"
    ),
}

DATASETS = [
    "Sentinel-2",
    "Sentinel-1",
    "Landsat",
    "HLS",
]

PLOT_PROPERTY = "Name"

# Small geometric tolerance, not permission to monitor outside Texas.
OUTSIDE_AREA_TOLERANCE_M2 = 10.0


# --------------------------------------------------
# Credentials and Earth Engine initialization
# --------------------------------------------------
def read_secrets():
    """Return Streamlit secrets, or an empty dict if no file exists."""
    try:
        return st.secrets.to_dict()
    except FileNotFoundError:
        return {}

from google.oauth2 import service_account

@st.cache_resource
def initialize_ee():
    secrets = read_secrets()
    project = secrets.get("project", GEE_PROJECT)

    scopes = [
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/devstorage.full_control",
    ]

    if "gee" in secrets:
        cred_dict = dict(secrets["gee"])
        
        # Ensure literal '\n' string escapes are converted to valid multiline breaks
        if "private_key" in cred_dict:
            cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")

        credentials = service_account.Credentials.from_service_account_info(
            cred_dict, 
            scopes=scopes
        )

    elif LOCAL_KEY_FILE.is_file():
        credentials = service_account.Credentials.from_service_account_file(
            str(LOCAL_KEY_FILE), 
            scopes=scopes
        )

    else:
        raise FileNotFoundError(
            "No Earth Engine credentials were found. "
            "Configure a [gee] section in Streamlit secrets."
        )

    # Initialize Earth Engine with the structured credentials object
    ee.Initialize(credentials=credentials, project=project)

    return project

try:
    active_project = initialize_ee()
except Exception as exc:
    st.error(f"Earth Engine initialization failed: {exc}")
    st.stop()


# --------------------------------------------------
# Texas boundary and field selection
# --------------------------------------------------
def texas_geometry():
    return (
        ee.FeatureCollection("TIGER/2018/States")
        .filter(ee.Filter.eq("NAME", "Texas"))
        .geometry()
    )


def texas_features(asset):
    return (
        ee.FeatureCollection(asset)
        .filterBounds(texas_geometry())
        .filter(ee.Filter.notNull([PLOT_PROPERTY]))
    )


@st.cache_data(
    ttl=24 * 3600,
    show_spinner="Reading Texas plots ...",
)
def field_names(asset):
    values = (
        texas_features(asset)
        .aggregate_array(PLOT_PROPERTY)
        .distinct()
        .getInfo()
    )

    return sorted(values, key=str)


def selected_feature(asset, plot):
    """Select one uniquely named feature intersecting Texas."""
    matches = texas_features(asset).filter(
        ee.Filter.eq(PLOT_PROPERTY, plot)
    )

    count = matches.size().getInfo()

    if count == 0:
        raise ValueError(
            f"No Texas feature found with "
            f"{PLOT_PROPERTY}={plot!r}."
        )

    if count > 1:
        raise ValueError(
            f"{count} Texas features share "
            f"{PLOT_PROPERTY}={plot!r}. "
            "Give them unique names or intentionally combine "
            "their boundaries before monitoring."
        )

    return ee.Feature(matches.first())


@st.cache_data(
    ttl=24 * 3600,
    show_spinner="Checking field boundary ...",
)
def boundary_info(asset, plot):
    feature = selected_feature(asset, plot)
    geom = feature.geometry()

    geometry_type = geom.type().getInfo()

    if geometry_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            "Select a Polygon or MultiPolygon boundary. "
            f"The selected feature is {geometry_type}."
        )

    area_m2 = geom.area(1).getInfo()

    if area_m2 <= 0:
        raise ValueError("The selected polygon has no usable area.")

    outside_m2 = (
        geom.difference(texas_geometry(), 1)
        .area(1)
        .getInfo()
    )

    if outside_m2 > OUTSIDE_AREA_TOLERANCE_M2:
        raise ValueError(
            "This application monitors Texas fields only. "
            f"The selected boundary has approximately "
            f"{outside_m2 / 10000:.3f} ha outside Texas. "
            "Review the boundary before continuing."
        )

    core = geom.buffer(ft.EDGE_BUFFER_M, 1)
    core_m2 = core.area(1).getInfo()

    if core_m2 <= 0:
        raise ValueError(
            "No analysis area remains after the inward buffer. "
            "Check the boundary or adjust EDGE_BUFFER_M."
        )

    qc = ft.boundary_qc(geom)
    qc["outside_texas_m2"] = round(outside_m2, 2)

    return qc, geom.getInfo()


# --------------------------------------------------
# Dataset helpers
# --------------------------------------------------
@st.cache_data(
    ttl=3600,
    show_spinner="Finding Sentinel-1 relative orbits ...",
)
def available_s1_orbits(
    asset,
    plot,
    start,
    end_exclusive,
    orbit_pass,
):
    geom = selected_feature(asset, plot).geometry()

    return sat.s1_orbits(
        geom,
        start,
        end_exclusive,
        orbit_pass,
    )


@st.cache_data(
    ttl=24 * 3600,
    show_spinner="Reading satellite observations ...",
)
def load_series(
    asset,
    plot,
    start,
    end_exclusive,
    source,
    variable,
    orbit_pass,
    relative_orbit,
):
    geom = selected_feature(asset, plot).geometry()

    if source == "Sentinel-2":
        return ft.extract_series(
            geom,
            start,
            end_exclusive,
        )

    return sat.extract_series(
        geom,
        start,
        end_exclusive,
        source,
        variable,
        orbit_pass,
        relative_orbit,
    )


@st.cache_data(
    ttl=3600,
    show_spinner="Loading satellite image layers ...",
)
def image_tiles(
    asset,
    plot,
    date_str,
    source,
    orbit_pass,
    relative_orbit,
):
    if source == "Sentinel-2":
        return ft.s2_tile_urls(
            asset,
            plot,
            date_str,
        )

    return sat.tile_urls(
        asset,
        plot,
        date_str,
        source,
        orbit_pass,
        relative_orbit,
    )


def widget_key(prefix, context):
    digest = hashlib.sha256(
        repr(context).encode("utf-8")
    ).hexdigest()[:20]

    return f"{prefix}_{digest}"


# --------------------------------------------------
# User interface
# --------------------------------------------------
st.title("Texas Field Practice Monitor")

st.caption(
    "Select a Texas field and dataset. "
    "Click a time-series point to view that date's imagery."
)

with st.sidebar:
    st.header("Monitoring settings")
    st.caption("Region: Texas")

    farm = st.selectbox(
        "Farm / field asset",
        list(ASSETS),
    )
    asset = ASSETS[farm]

    try:
        names = field_names(asset)
    except Exception as exc:
        st.error(f"Could not read the asset: {exc}")
        st.stop()

    if not names:
        st.warning(
            "No named features intersecting Texas were found "
            "in this asset."
        )
        st.stop()

    plot = st.selectbox("Plot / management area", names)

    source = st.selectbox("Dataset", DATASETS)

    today = dt.date.today()

    start = st.date_input(
        "Start date",
        value=dt.date(2025, 1, 1),
        max_value=today,
    )

    end = st.date_input(
        "End date",
        value=today,
        max_value=today,
    )

    if start > end:
        st.error("Start date must not be after end date.")
        st.stop()

    # The interface uses inclusive dates.
    # Earth Engine receives an exclusive end date.
    end_exclusive = end + dt.timedelta(days=1)


# Validate the boundary before searching for imagery.
try:
    qc, geojson = boundary_info(asset, plot)
except Exception as exc:
    st.error(f"Boundary check failed: {exc}")
    st.stop()


with st.sidebar:
    orbit_pass = "ASCENDING"
    relative_orbit = None

    if source == "Sentinel-1":
        variable = st.radio(
            "Radar polarization",
            ["VV", "VH"],
            horizontal=True,
        )

        orbit_pass = st.selectbox(
            "Orbit direction",
            ["ASCENDING", "DESCENDING"],
        )

        try:
            orbits = available_s1_orbits(
                asset,
                plot,
                str(start),
                str(end_exclusive),
                orbit_pass,
            )
        except Exception as exc:
            st.error(f"Could not list radar orbits: {exc}")
            st.stop()

        if not orbits:
            st.info(
                "No observations were found for this orbit "
                "direction. Try the other direction or "
                "a longer date range."
            )
            st.stop()

        relative_orbit = st.selectbox(
            "Relative orbit",
            orbits,
        )

        st.caption(
            "The radar series uses one orbit direction "
            "and one relative orbit."
        )

    else:
        variable = "NDVI"
        st.caption("Variable: NDVI")

    series_choice = st.multiselect(
        "Series to display",
        ["Raw", "Spike-cleaned"],
        default=["Raw", "Spike-cleaned"],
    )

    reported = st.date_input(
        "Farmer-reported practice date (optional)",
        value=None,
    )

    if st.button("Refresh cached field data"):
        st.cache_data.clear()
        st.rerun()


# --------------------------------------------------
# Retrieve time series
# --------------------------------------------------
try:
    series = load_series(
        asset,
        plot,
        str(start),
        str(end_exclusive),
        source,
        variable,
        orbit_pass,
        relative_orbit,
    )
except Exception as exc:
    st.error(f"Satellite extraction failed: {exc}")
    st.stop()


# Reset selected image date when the monitoring context changes.
context = (
    asset,
    plot,
    str(start),
    str(end),
    source,
    variable,
    orbit_pass,
    relative_orbit,
)

if st.session_state.get("monitor_context") != context:
    st.session_state["monitor_context"] = context
    st.session_state["picked_date"] = None

map_column, chart_column = st.columns([1, 1])

# --------------------------------------------------
# Time series: render first to receive chart selection
# --------------------------------------------------
with chart_column:
    st.subheader(f"{source}: {variable} time series")
    if series.empty:
        st.info(
            "No usable observations passed the current "
            "coverage and quality filters."
        )

    elif not series_choice:
        st.info("Select at least one series to display.")

    elif variable not in series.columns:
        st.error(f"The extracted table does not contain {variable!r}.")

    else:
        data = series.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = ft.clean_series(data, variable)

        labels = {
            variable: "Raw",
            f"{variable}_clean": "Spike-cleaned",
        }
        columns_to_plot = [
            col for col, label in labels.items() if label in series_choice
        ]

        long = pd.melt(
            data,
            id_vars="date",
            value_vars=columns_to_plot,
            var_name="series",
            value_name="value",
        )
        long["series"] = long["series"].map(labels)
        long["date_str"] = long["date"].dt.strftime("%Y-%m-%d")

        point_selection = alt.selection_point(
            name="pick",
            fields=["date_str"],
            toggle=False,
        )

        axis_title = (
            f"{variable}: mean backscatter (dB)"
            if source == "Sentinel-1"
            else "NDVI"
        )

        base = alt.Chart(long).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title=axis_title),
            color=alt.Color(
                "series:N",
                title="Series",
                # Fixed colors so each series keeps its color when toggled
                scale=alt.Scale(
                    domain=["Raw", "Spike-cleaned"],
                    range=["#1f77b4", "#ff7f0e"],
                ),
            ),
        )

        line = base.mark_line()

        points = (
            base.mark_point(filled=True, size=75)
            .encode(
                opacity=(
                    alt.when(point_selection)
                    .then(alt.value(1.0))
                    .otherwise(alt.value(0.35))
                ),
                tooltip=[
                    alt.Tooltip("date_str:N", title="Date"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("value:Q", title=variable, format=".3f"),
                ],
            )
            .add_params(point_selection)
        )

        chart = line + points

        if reported is not None:
            rule = (
                alt.Chart(pd.DataFrame({"date": [pd.Timestamp(reported)]}))
                .mark_rule(color="orange", strokeDash=[4, 4])
                .encode(x="date:T")
            )
            chart = chart + rule

        event = st.altair_chart(
            chart.properties(height=450),
            width="stretch",
            on_select="rerun",
            selection_mode="pick",
            key=widget_key("chart", (context, tuple(series_choice))),
        )

        selection = event.selection if event else None
        pick = selection.get("pick") if selection else None

        if pick:
            row = pick[0] if isinstance(pick, list) else pick
            st.session_state["picked_date"] = row.get("date_str")
# --------------------------------------------------
# Boundary and selected-date map
# --------------------------------------------------
picked = st.session_state.get("picked_date")

with map_column:
    st.subheader("Field boundary and imagery")

    st.write(
        f"Area: {qc['area_ha']} ha | "
        f"Analysis core: {qc['core_ha']} ha"
    )

    for flag in qc["flags"]:
        st.warning(flag)

    map_object = folium.Map(
        location=[31.0, -99.0],
        zoom_start=6,
        tiles=None,
    )

    folium.TileLayer(
        "Esri.WorldImagery",
        name="Satellite basemap",
    ).add_to(map_object)

    if picked:
        try:
            urls = image_tiles(
                asset,
                plot,
                picked,
                source,
                orbit_pass,
                relative_orbit,
            )
        except Exception as exc:
            urls = {}
            st.error(f"Could not load image layers: {exc}")

        if urls:
            default_layer = (
                f"{variable} (dB)"
                if source == "Sentinel-1"
                else "True color"
            )

            for name, url in urls.items():
                folium.TileLayer(
                    tiles=url,
                    attr=(
                        "Google Earth Engine / "
                        "Copernicus / USGS / NASA"
                    ),
                    name=f"{picked}: {name}",
                    overlay=True,
                    show=(name == default_layer),
                    max_zoom=20,
                ).add_to(map_object)

            st.caption(
                f"{source} imagery: {picked}. "
                "Use the layer control to switch views."
            )

        else:
            st.warning(
                "No image layers are available for the "
                "selected date and acquisition settings."
            )

    else:
        st.caption(
            "Click a point in the time series to load "
            "that date's image."
        )

    boundary = folium.GeoJson(
        geojson,
        name="Selected field boundary",
        style_function=lambda feature: {
            "color": "red",
            "weight": 2,
            "fill": False,
        },
    ).add_to(map_object)

    map_object.fit_bounds(boundary.get_bounds())

    folium.LayerControl(
        collapsed=False,
    ).add_to(map_object)

    st_folium(
        map_object,
        height=650,
        use_container_width=True,
        returned_objects=[],
        key=widget_key(
            "map",
            (
                asset,
                plot,
                source,
                variable,
                orbit_pass,
                relative_orbit,
                picked,
            ),
        ),
    )


# --------------------------------------------------
# Interpretation notes
# --------------------------------------------------
with st.expander("How to interpret these observations"):
    st.markdown(
        """
- Monitoring is restricted to Texas polygon boundaries.
- The chart summarizes the inward-buffered analysis core.
- Map layers are clipped to the original selected boundary.
- Same-day mosaics and daily averages are different products;
  map values need not exactly equal chart values.
- Optical datasets use their respective quality masks.
- Sentinel-1 uses the selected orbit direction and relative orbit.
- Temporal spike replacement is optional and can alter real events.
- HLS contains Landsat- and Sentinel-2-derived observations;
  it is not an independent source of additional satellites.
- A farmer-reported practice date is an annotation, not verification.
        """
    )