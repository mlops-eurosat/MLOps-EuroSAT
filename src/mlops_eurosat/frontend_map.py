"""Interactive map frontend for live EuroSAT land-use classification.

The red rectangle on the map is a live Leaflet overlay — it follows the map
centre in real-time and always shows the exact 640 m patch that will be sent
to the model. Classification runs automatically when the map stands still.

Run locally (set SH_CLIENT_ID and SH_CLIENT_SECRET first):
    streamlit run src/mlops_eurosat/frontend_map.py
"""

from __future__ import annotations

import base64
import io
import math
import os
import time

import folium
import requests
import streamlit as st
from folium.elements import MacroElement
from jinja2 import Template
from PIL import Image
from streamlit_folium import st_folium

API_URL = os.environ.get("API_URL", "https://eurosat-api-999981877996.europe-west3.run.app/predict")
SH_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

ZOOM = 15
DEBOUNCE_S = 1.0
DEFAULT_LAT, DEFAULT_LNG = 48.137, 11.575  # Munich
_PATCH_M = 640

_SH_EVALSCRIPT = """
//VERSION=3
function setup() {
    return { input: [{ bands: ["B04", "B03", "B02"] }], output: { bands: 3 } };
}
function evaluatePixel(s) {
    return [3.5 * s.B04, 3.5 * s.B03, 3.5 * s.B02];
}
"""


# ── Sentinel Hub auth ─────────────────────────────────────────────────────────

_sh_token: str | None = None
_sh_token_expiry: float = 0.0


def _sh_access_token() -> str:
    global _sh_token, _sh_token_expiry
    if _sh_token and time.time() < _sh_token_expiry - 30:
        return _sh_token
    r = requests.post(
        SH_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": os.environ["SH_CLIENT_ID"],
            "client_secret": os.environ["SH_CLIENT_SECRET"],
        },
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    _sh_token = data["access_token"]
    _sh_token_expiry = time.time() + data.get("expires_in", 600)
    return _sh_token


# ── patch fetching ────────────────────────────────────────────────────────────


def get_patch(lat: float, lng: float) -> Image.Image:
    """Return a 64×64 Sentinel-2 RGB patch (640 m) centred at (lat, lng)."""
    lat_delta = _PATCH_M / 111_320
    lng_delta = _PATCH_M / (111_320 * math.cos(math.radians(lat)))
    bbox = [lng - lng_delta / 2, lat - lat_delta / 2, lng + lng_delta / 2, lat + lat_delta / 2]

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"maxCloudCoverage": 30}}],
        },
        "output": {
            "width": 64,
            "height": 64,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
        "evalscript": _SH_EVALSCRIPT,
    }
    headers = {"Authorization": f"Bearer {_sh_access_token()}", "Content-Type": "application/json"}
    r = requests.post(SH_PROCESS_URL, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


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
        location=[DEFAULT_LAT, DEFAULT_LNG], zoom_start=ZOOM, zoom_control=False, scrollWheelZoom=False
    )
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
    "Pan the map — the **red rectangle** follows the map centre and shows "
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
            st.image(st.session_state.patch, caption="64×64 Sentinel-2 patch sent to model (~640 m)", width=192)

        st.divider()
        st.subheader("Class probabilities")

        import pandas as pd

        df = (
            pd.DataFrame(pred["probabilities"].items(), columns=["Class", "Probability"])
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(df.style.format({"Probability": "{:.2%}"}), hide_index=True, use_container_width=True)
