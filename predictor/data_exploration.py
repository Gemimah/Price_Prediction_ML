import copy
import json
import folium
import requests
from pathlib import Path
from shapely.geometry import shape
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

_rwanda_geojson_cache = None

# geoBoundaries REST API — returns metadata including the actual GeoJSON download URL
RWANDA_GEOBOUNDARIES_API = (
    "https://www.geoboundaries.org/api/current/gbOpen/RWA/ADM2/"
)


def _get_rwanda_geojson():
    global _rwanda_geojson_cache
    if _rwanda_geojson_cache is not None:
        return _rwanda_geojson_cache

    cache_path = BASE_DIR / "dummy-data" / "rwanda_districts.geojson"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            _rwanda_geojson_cache = json.load(f)
        return _rwanda_geojson_cache

    # Step 1: ask the API for the real download URL
    api_resp = requests.get(RWANDA_GEOBOUNDARIES_API, timeout=20)
    api_resp.raise_for_status()
    if not api_resp.text.strip():
        raise ValueError("geoBoundaries API returned an empty response.")
    api_data = api_resp.json()

    # Prefer simplified geometry; fall back to full geometry
    geojson_url = api_data.get("sjDownloadURL") or api_data.get("gjDownloadURL")
    if not geojson_url:
        raise ValueError(f"No download URL in API response: {api_data}")

    # Step 2: download the actual GeoJSON
    resp = requests.get(geojson_url, timeout=30)
    resp.raise_for_status()
    if not resp.text.strip():
        raise ValueError(f"GeoJSON download from {geojson_url} returned an empty response.")
    data = resp.json()

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    _rwanda_geojson_cache = data
    return data


def generate_rwanda_district_map(df):
    district_counts = df.groupby("district").size().reset_index(name="client_count")
    district_dict = {
        name.strip().lower(): int(count)
        for name, count in zip(district_counts["district"], district_counts["client_count"])
    }

    geojson = copy.deepcopy(_get_rwanda_geojson())

    # Inject client_count into each feature so tooltip can display it
    for feature in geojson["features"]:
        raw_name = feature["properties"].get("shapeName", "")
        feature["properties"]["client_count"] = district_dict.get(raw_name.strip().lower(), 0)

    m = folium.Map(location=[-1.9403, 29.8739], zoom_start=8, tiles="CartoDB positron")

    choropleth = folium.Choropleth(
        geo_data=geojson,
        name="Vehicle Clients per District",
        data=district_counts,
        columns=["district", "client_count"],
        key_on="feature.properties.shapeName",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.6,
        legend_name="Number of Vehicle Clients",
        nan_fill_color="#dddddd",
        highlight=True,
    ).add_to(m)

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=["shapeName", "client_count"],
            aliases=["District:", "Vehicle Clients:"],
            style=(
                "font-size:13px; font-family:Arial; background:white;"
                " padding:6px; border-radius:4px; box-shadow:2px 2px 4px rgba(0,0,0,.2);"
            ),
        )
    )

    # District name + count labels at centroids
    for feature in geojson["features"]:
        name = feature["properties"].get("shapeName", "")
        count = feature["properties"]["client_count"]
        try:
            geom = shape(feature["geometry"])
            centroid = geom.centroid
            label_html = (
                f'<div style="font-size:8px;color:#1a1a1a;font-weight:bold;'
                f"text-align:center;white-space:nowrap;pointer-events:none;"
                f'text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff;">'
                f"{name}<br>"
                f'<span style="color:#c0392b;font-size:9px;">{count} clients</span>'
                f"</div>"
            )
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(
                    html=label_html,
                    icon_size=(110, 32),
                    icon_anchor=(55, 16),
                ),
            ).add_to(m)
        except Exception:
            pass

    folium.LayerControl().add_to(m)
    return m._repr_html_()


# Data Exploration
def dataset_exploration(df):
 table_html = df.head().to_html(
 classes="table table-bordered table-striped table-sm",
 float_format="%.2f",
 justify="center",
 index=False,
 )
 return table_html

# Data description
def data_exploration(df):
 table_html = df.head().to_html(
 classes="table table-bordered table-striped table-sm",
 float_format="%.2f",
 justify="center",
 )
 return table_html












