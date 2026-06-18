"""Interactive map frontend for live EuroSAT land-use classification.

The red rectangle on the map is a live Leaflet overlay — it follows the map
centre in real-time and always shows the exact 640 m patch that will be sent
to the model. Classification runs automatically when the map stands still.

Run locally:
    streamlit run src/mlops_eurosat/frontend_map.py
"""

from __future__ import annotations

import base64
import io
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import folium
import requests
import streamlit as st
from folium.elements import MacroElement
from jinja2 import Template
from PIL import Image
from streamlit_folium import st_folium

API_URL = os.environ.get("API_URL", "https://eurosat-api-999981877996.europe-west3.run.app/predict")

ZOOM = 15
TILE_PX = 256
# Equatorial ground resolution at zoom 15.
# Actual m/px at latitude φ = M_PER_PX * cos(φ).
M_PER_PX = 40_075_016.7 / (TILE_PX * 2**ZOOM)

ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Tiles &copy; Esri &mdash; Esri, Maxar, Earthstar Geographics"
DEBOUNCE_S = 1.0

DEFAULT_LAT, DEFAULT_LNG = 48.137, 11.575  # Munich


# ── tile helpers ─────────────────────────────────────────────────────────────


def _tile_coords(lat: float, lng: float) -> tuple[int, int, int, int]:
    n = 2**ZOOM
    lat_r = math.radians(lat)
    fx = (lng + 180) / 360 * n
    fy = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
    tx, ty = int(fx), int(fy)
    px, py = int((fx - tx) * TILE_PX), int((fy - ty) * TILE_PX)
    return tx, ty, px, py


def _fetch_tile(tx: int, ty: int) -> tuple[int, int, Image.Image]:
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{ZOOM}/{ty}/{tx}"
    r = requests.get(url, timeout=10, headers={"User-Agent": "EuroSAT-Map/1.0"})
    r.raise_for_status()
    return tx, ty, Image.open(io.BytesIO(r.content)).convert("RGB")


def get_patch(lat: float, lng: float) -> Image.Image:
    """Return a 64×64 satellite patch (640 m) centred at (lat, lng).

    Fetches a 3×3 tile grid in parallel to avoid black borders when the crop
    centre sits near a tile edge. Crop size is latitude-corrected.
    """
    # pixels needed to cover 640 m at this latitude
    crop_px = round(640 / (M_PER_PX * math.cos(math.radians(lat))))
    half = crop_px // 2

    tx, ty, px, py = _tile_coords(lat, lng)

    canvas = Image.new("RGB", (TILE_PX * 3, TILE_PX * 3))
    offsets = [(dx, dy) for dy in range(-1, 2) for dx in range(-1, 2)]

    with ThreadPoolExecutor(max_workers=9) as pool:
        futures = [pool.submit(_fetch_tile, tx + dx, ty + dy) for dx, dy in offsets]
        for f in as_completed(futures):
            try:
                ftx, fty, tile = f.result()
                canvas.paste(tile, ((ftx - tx + 1) * TILE_PX, (fty - ty + 1) * TILE_PX))
            except Exception:
                pass  # leave black if a tile fails (e.g. ocean, polar regions)

    cx, cy = TILE_PX + px, TILE_PX + py
    patch = canvas.crop((cx - half, cy - half, cx + half, cy + half))
    return patch.resize((64, 64), Image.Resampling.LANCZOS)


# ── API ───────────────────────────────────────────────────────────────────────


def classify(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode()
    r = requests.post(API_URL, json={"instances": [{"image_b64": image_b64}]}, timeout=30)
    r.raise_for_status()
    return r.json()["predictions"][0]


# ── map builder ───────────────────────────────────────────────────────────────


class _LiveCenterRect(MacroElement):
    """Overlay div appended directly to map.getContainer() — outside Leaflet's pane
    system, so tiles can never render above it regardless of scroll or tile load order."""

    def __init__(self) -> None:
        super().__init__()
        self._name = "LiveCenterRect"
        self._template = Template("""
            {% macro script(this, kwargs) %}
            (function () {
                var map = {{ this._parent.get_name() }};

                // Append directly to the Leaflet container div, not to any pane.
                var el = document.createElement('div');
                el.style.cssText = [
                    'position:absolute',
                    'top:50%',
                    'left:50%',
                    'transform:translate(-50%,-50%)',
                    'z-index:9999',
                    'pointer-events:none',
                    'border:3px solid #ff4b4b',
                    'border-radius:4px',
                    'box-shadow:0 0 0 2px rgba(255,255,255,0.4)'
                ].join(';');
                map.getContainer().appendChild(el);

                function update() {
                    var zoom = map.getZoom();
                    var lat  = map.getCenter().lat;
                    var mPerPx = 40075016.7 / (256 * Math.pow(2, zoom))
                                 * Math.cos(lat * Math.PI / 180);
                    var px = Math.round(640 / mPerPx);
                    el.style.width  = px + 'px';
                    el.style.height = px + 'px';
                }

                map.on('move zoomend', update);
                update();
            })();
            {% endmacro %}
        """)


@st.cache_resource
def _base_map() -> folium.Map:
    """Cached once per session — same object → same HTML → iframe is never recreated on rerun."""
    m = folium.Map(
        location=[DEFAULT_LAT, DEFAULT_LNG], zoom_start=ZOOM, tiles=None, zoom_control=False, scrollWheelZoom=False
    )
    folium.TileLayer(tiles=ESRI_TILES, attr=ESRI_ATTR, name="Satellite").add_to(m)
    _LiveCenterRect().add_to(m)
    return m


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="EuroSAT Live Map", layout="wide")
st.markdown(
    """
<style>
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] { flex-direction: column; }
    [data-testid="column"] { width: 100% !important; min-width: 100% !important; }
    iframe { height: 350px !important; }
}
</style>
""",
    unsafe_allow_html=True,
)
st.title("EuroSAT Live Land Use Classification")
st.caption(
    "Pan the satellite map — the **red rectangle** follows the map centre and shows "
    "exactly the 640 m patch the model will classify. Results update automatically."
)

for key, default in [
    ("last_inferred", None),
    ("result", None),
    ("patch", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

col_map, col_result = st.columns([3, 2])

with col_map:
    map_data = st_folium(_base_map(), use_container_width=True, height=520, returned_objects=["center"])

if map_data and map_data.get("center"):
    new_center = (map_data["center"]["lat"], map_data["center"]["lng"])
    if new_center != st.session_state.last_inferred:
        time.sleep(DEBOUNCE_S)
        with col_result:
            with st.spinner("Running inference…"):
                try:
                    patch = get_patch(*new_center)
                    pred = classify(patch)
                    st.session_state.patch = patch
                    st.session_state.result = pred
                    st.session_state.last_inferred = new_center
                except Exception as e:
                    st.session_state.result = {"error": str(e)}
                    st.session_state.patch = None
        # No st.rerun() — results render in this same run, map iframe stays alive.

with col_result:
    if st.session_state.result is None:
        st.info("Pan the map to classify a location.")
    elif "error" in st.session_state.result:
        st.error(f"Error: {st.session_state.result['error']}")
    else:
        pred = st.session_state.result
        lat, lng = st.session_state.last_inferred

        st.metric("Location", f"{lat:.5f}, {lng:.5f}")
        st.metric("Predicted class", pred["class_name"])
        top_prob = max(pred["probabilities"].values())
        st.metric("Confidence", f"{top_prob * 100:.1f}%")

        if st.session_state.patch:
            st.image(st.session_state.patch, caption="64×64 patch sent to model (~640 m)", width=192)

        st.divider()
        st.subheader("Class probabilities")

        import pandas as pd

        df = (
            pd.DataFrame(pred["probabilities"].items(), columns=["Class", "Probability"])
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(df.style.format({"Probability": "{:.2%}"}), hide_index=True, use_container_width=True)
