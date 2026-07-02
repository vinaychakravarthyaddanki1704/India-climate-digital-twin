import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob, os
import geopandas as gpd
from shapely.ops import unary_union
from shapely import prepared
from shapely.geometry import Point

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Climate Digital Twin | NRSC/ISRO",
    layout="wide", page_icon="🌏",
    initial_sidebar_state="expanded"
)

# ── Dark CSS (DestinE-style) ───────────────────────────────────────────────────
st.markdown("""<style>
.stApp{background:#030b1a;color:#c8d8e8;}
[data-testid="stSidebar"]{background:#040d1a;border-right:1px solid #1a3a5c;}
[data-testid="stSidebar"] p{color:#7ab8d4;}
.stRadio label{color:#a0c8e0!important;}
.stSelectbox label,.stSlider label{color:#7ab8d4!important;}
[data-testid="stMetric"]{background:linear-gradient(135deg,#0a1628,#0d2040);
  border:1px solid #1a3a5c;border-radius:12px;padding:1rem;}
[data-testid="stMetricLabel"] p{color:#7ab8d4!important;font-size:0.78rem!important;
  text-transform:uppercase;letter-spacing:1px;}
[data-testid="stMetricValue"]{color:#00c8ff!important;}
[data-testid="stMetricDelta"]{color:#00e676!important;}
#MainMenu,footer,header{visibility:hidden;}
div[data-testid="stDecoration"]{display:none;}
.block-container{padding-top:0.5rem!important;}
hr{border-color:#1a3a5c;}
.stSelectbox>div>div{background:#051020;border:1px solid #1a3a5c;color:#c8d8e8;}
.stTextInput>div>div>input{background:#051020;border:1px solid #1a3a5c;color:#c8d8e8;}
</style>""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = "data" if os.path.exists("data") else "/mnt/d/NRSC Internship"
GC_TEMP  = f"{BASE}/graphcast/india_outputs"
CNN_TEMP = f"{BASE}/cnn-correction/cnn-correction/corrected_outputs"
IMD_TEMP = f"{BASE}/cnn-correction/cnn-correction/data/imd_monthly"
GC_RAIN  = f"{BASE}/graphcast/india_rainfall"
CNN_RAIN = f"{BASE}/cnn-correction/cnn-correction/corrected_rainfall"
IMD_RAIN = f"{BASE}/cnn-correction/cnn-correction/data/imd_rainfall_monthly"
SHP_PATH = f"{BASE}/cnn-correction/cnn-correction/data/STATE_BDY_UPD/STATE_BDY_UPD.shp"

# ── Grid ───────────────────────────────────────────────────────────────────────
LAT = np.linspace(6.50, 38.50, 129)
LON = np.linspace(66.50, 100.50, 137)
LON_2D, LAT_2D = np.meshgrid(LON, LAT)
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

RAIN_CS = [[0.00,'rgb(255,255,255)'],[0.10,'rgb(0,0,200)'],
           [0.30,'rgb(0,180,255)'],  [0.50,'rgb(0,180,0)'],
           [0.70,'rgb(255,255,0)'],  [0.85,'rgb(255,140,0)'],
           [1.00,'rgb(200,0,0)']]
TEMP_CS = [[0,'rgb(0,0,180)'],[0.25,'rgb(0,100,255)'],
           [0.5,'rgb(255,255,100)'],[0.75,'rgb(255,80,0)'],
           [1,'rgb(180,0,0)']]
RDBU_CS = [[0,'rgb(0,80,200)'],[0.5,'rgb(255,255,255)'],[1,'rgb(200,0,0)']]

cmap_rain = mcolors.LinearSegmentedColormap.from_list(
    'rain', ['white','blue','deepskyblue','green','yellow','orange','red'])

# ── Masks (cached) ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_masks():
    first = sorted(glob.glob(f"{GC_RAIN}/*.npy"))[0]
    LAND  = np.isfinite(np.load(first)[0])
    _gdf  = gpd.read_file(SHP_PATH)
    _geom = unary_union(_gdf.geometry)
    _prep = prepared.prep(_geom)
    pts   = [Point(lo, la) for la, lo in zip(LAT_2D.ravel(), LON_2D.ravel())]
    INDIA_MASK = np.array([_prep.contains(p) for p in pts]).reshape(LAT_2D.shape)
    INDIA = LAND & INDIA_MASK
    return LAND, INDIA, _gdf

with st.spinner("🌏 Initialising India domain..."):
    LAND, INDIA, _gdf = load_masks()

def msk(arr):
    a = arr.copy().astype(float); a[~INDIA] = np.nan; return a

# ── Plotly helpers ─────────────────────────────────────────────────────────────
def globe(data_dict, vmax, title="", label="mm/day", cs=RAIN_CS, vmin=0):
    traces = []
    show_bar = True
    for name, arr in data_dict.items():
        v = np.isfinite(arr)
        traces.append(go.Scattergeo(
            lat=LAT_2D[v].ravel(), lon=LON_2D[v].ravel(),
            mode='markers', name=name,
            marker=dict(size=5, color=arr[v].ravel(),
                        colorscale=cs, cmin=vmin, cmax=vmax,
                        colorbar=dict(title=dict(text=label,side='right',
                                                  font=dict(color='#7ab8d4')),
                                      thickness=15, len=0.7,
                                      bgcolor='rgba(3,11,26,0.8)',
                                      tickfont=dict(color='#7ab8d4'),
                                      bordercolor='#1a3a5c'),
                        showscale=show_bar, opacity=0.9),
            showlegend=len(data_dict)>1))
        show_bar = False
    fig = go.Figure(traces)
    fig.update_geos(projection_type="orthographic",
                    projection_rotation=dict(lon=82,lat=22,roll=0),
                    bgcolor="rgba(3,11,26,0)",
                    showland=True,  landcolor="#0d1f38",
                    showocean=True, oceancolor="#030b1a",
                    showcoastlines=True, coastlinecolor="#2a5a8a",
                    showcountries=True, countrycolor="#1a4a7a",
                    showframe=False)
    fig.update_layout(
        title=dict(text=title, font=dict(color='#00c8ff',size=14), x=0.5, xanchor='center', pad=dict(t=8)),
        paper_bgcolor="#030b1a", margin=dict(l=0,r=0,t=55,b=0), height=500,
        legend=dict(bgcolor='rgba(3,11,26,0.9)',bordercolor='#1a3a5c',
                    font=dict(color='#7ab8d4')))
    return fig

def surface3d(arr, vmax, title="", label="mm/day", cs=RAIN_CS, vmin=0):
    z = arr.copy(); z[~INDIA] = 0; z[~np.isfinite(z)] = 0
    fig = go.Figure(go.Surface(
        z=z, x=LON, y=LAT, colorscale=cs, cmin=vmin, cmax=vmax,
        colorbar=dict(title=dict(text=label,side='right',
                                  font=dict(color='#7ab8d4')),
                      thickness=15, len=0.7,
                      bgcolor='rgba(3,11,26,0.8)',
                      tickfont=dict(color='#7ab8d4'),
                      bordercolor='#1a3a5c'),
        lighting=dict(ambient=0.6,diffuse=0.8,specular=0.3,roughness=0.5)))
    fig.update_layout(
        title=dict(text=title, font=dict(color='#00c8ff',size=14), x=0.5, xanchor='center', pad=dict(t=8)),
        scene=dict(
            xaxis=dict(title=dict(text='Lon (°E)', font=dict(color='#7ab8d4')),
                       tickfont=dict(color='#7ab8d4'), gridcolor='#1a3a5c',
                       backgroundcolor='#030b1a', range=[66.5,100.5]),
            yaxis=dict(title=dict(text='Lat (°N)', font=dict(color='#7ab8d4')),
                       tickfont=dict(color='#7ab8d4'), gridcolor='#1a3a5c',
                       backgroundcolor='#030b1a', range=[6.5,38.5]),
            zaxis=dict(title=dict(text=label, font=dict(color='#7ab8d4')),
                       tickfont=dict(color='#7ab8d4'), gridcolor='#1a3a5c',
                       backgroundcolor='#030b1a'),
            bgcolor='#030b1a',
            camera=dict(eye=dict(x=1.8,y=-1.8,z=1.4))),
        paper_bgcolor='#030b1a', margin=dict(l=0,r=0,t=55,b=0), height=500)
    return fig

def map2d_plotly(data_dict, vmax, title="", label="mm/day", cs=RAIN_CS, vmin=0):
    """Plotly flat 2D Mercator map of India — dark DestinE theme."""
    traces = []
    show_bar = True
    for name, arr in data_dict.items():
        v = np.isfinite(arr)
        traces.append(go.Scattergeo(
            lat=LAT_2D[v].ravel(), lon=LON_2D[v].ravel(),
            mode='markers', name=name,
            marker=dict(size=4, color=arr[v].ravel(),
                        colorscale=cs, cmin=vmin, cmax=vmax,
                        colorbar=dict(title=dict(text=label, side='right',
                                                  font=dict(color='#7ab8d4')),
                                      thickness=15, len=0.7,
                                      bgcolor='rgba(3,11,26,0.8)',
                                      tickfont=dict(color='#7ab8d4'),
                                      bordercolor='#1a3a5c'),
                        showscale=show_bar, opacity=0.95, symbol='square'),
            showlegend=len(data_dict) > 1))
        show_bar = False
    fig = go.Figure(traces)
    fig.update_geos(
        projection_type="mercator",
        lonaxis=dict(range=[66.5, 100.5]),
        lataxis=dict(range=[6.5, 38.5]),
        bgcolor="rgba(3,11,26,0)",
        showland=True,  landcolor="#0d1f38",
        showocean=True, oceancolor="#030b1a",
        showcoastlines=True, coastlinecolor="#2a5a8a",
        showcountries=True, countrycolor="#1a4a7a",
        showsubunits=True, subunitcolor="#1a3a5c",
        showframe=False)
    fig.update_layout(
        title=dict(text=title, font=dict(color='#00c8ff', size=13), x=0.5),
        paper_bgcolor="#030b1a",
        margin=dict(l=0, r=0, t=55, b=0), height=500,
        legend=dict(bgcolor='rgba(3,11,26,0.9)', bordercolor='#1a3a5c',
                    font=dict(color='#7ab8d4')))
    return fig

def map2d(data, vmax, title, cmap=cmap_rain, vmin=0, gdf_reg=None):
    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor('#030b1a'); ax.set_facecolor('#0d1f38')
    _gdf[_gdf['STATE']!='DISPUTED (JHARKHAND & BIHAR)'].boundary.plot(
        ax=ax, color='#2a5a8a', linewidth=0.5)
    im = ax.pcolormesh(LON_2D, LAT_2D, data, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='auto')
    if gdf_reg is not None:
        gdf_reg.boundary.plot(ax=ax, color='red', linewidth=2.0)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.1)
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.yaxis.label.set_color('#7ab8d4'); cb.ax.tick_params(colors='#7ab8d4')
    ax.set_title(title, color='#00c8ff', fontsize=9, fontweight='bold')
    ax.set_xlim(66.5,100.5); ax.set_ylim(6.5,38.5)
    ax.tick_params(colors='#7ab8d4')
    for sp in ax.spines.values(): sp.set_edgecolor('#1a3a5c')
    plt.tight_layout(); return fig

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#030b1a 0%,#061428 50%,#030b1a 100%);
            padding:1.8rem 2.5rem 1.2rem;border-bottom:2px solid #1a3a5c;
            border-radius:0 0 20px 20px;margin-bottom:1.2rem;">
  <div style="display:flex;align-items:center;gap:1.5rem;">
    <span style="font-size:3.5rem;">🌏</span>
    <div>
      <h1 style="margin:0;font-size:2.1rem;font-weight:900;
                 background:linear-gradient(90deg,#00c8ff,#0066ff,#00c8ff);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        India Climate Digital Twin
      </h1>
      <p style="margin:0.3rem 0 0;color:#5a8aaa;font-size:0.9rem;">
        NRSC / ISRO &nbsp;·&nbsp; GraphCast GNN &nbsp;+&nbsp; U-Net CNN Bias Correction
        &nbsp;·&nbsp; IMD RF25 Validation
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

avail_months = sorted([os.path.basename(f).replace('.npy','')
                        for f in glob.glob(f"{GC_RAIN}/*.npy")])
yr0 = avail_months[0][:4]  if avail_months else "?"
yr1 = avail_months[-1][:4] if avail_months else "?"

c1,c2,c3,c4 = st.columns(4)
c1.metric("📅 Data Span",       f"{yr0} – {yr1}")
c2.metric("📦 Monthly Files",   str(len(avail_months)))
c3.metric("📍 India Grid Pts",  f"{int(np.sum(INDIA)):,}")
c4.metric("🧠 AI Engine",       "GraphCast + U-Net")
st.markdown("<hr style='margin:0.5rem 0 1rem;'>", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
      <div style="font-size:2.5rem;">🌏</div>
      <div style="color:#00c8ff;font-weight:800;font-size:1.1rem;letter-spacing:1px;">
        CLIMATE TWIN</div>
      <div style="color:#3a6a8a;font-size:0.72rem;">NRSC / ISRO</div>
    </div>
    <hr style="border-color:#1a3a5c;margin:0.7rem 0;">
    """, unsafe_allow_html=True)

    mode = st.radio("", [
        "🌐  3D Globe / Hero",
        "📊  Rainfall Monitor",
        "🌡️  Temperature Monitor",
        "🌧️  Flood Event Replay",
        "🔮  What-If Scenario",
    ])

    st.markdown("""
    <hr style="border-color:#1a3a5c;margin:1.2rem 0 0.7rem;">
    <div style="color:#3a6a8a;font-size:0.72rem;line-height:1.8;">
      <b style="color:#5a9ab8;">Data Sources</b><br>
      ERA5 · ECMWF<br>IMD RF25 (0.25°)<br>GraphCast · DeepMind<br>Survey of India SHP
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 0 — 3D GLOBE HERO
# ══════════════════════════════════════════════════════════════════════════════
if "3D Globe" in mode:
    st.markdown("""<div style="color:#00c8ff;font-size:1.3rem;font-weight:700;
    border-left:4px solid #00c8ff;padding-left:1rem;margin-bottom:1rem;">
    🌐 Live 3D Globe — India Climate State</div>""", unsafe_allow_html=True)

    ca, cb_col = st.columns([3,1])
    with ca:
        sel = st.selectbox("Month", avail_months, index=len(avail_months)-1)
        yr, mo = int(sel[:4]), int(sel[5:])
    with cb_col:
        vtype = st.radio("View", ["🌍 Globe","🗺️ 2D Map","⛰️ 3D Surface"], label_visibility="collapsed")

    gc_r = msk(np.nanmean(np.load(f"{GC_RAIN}/{sel}.npy"), 0))
    cnn_p = f"{CNN_RAIN}/{sel}.npy"
    cnn_r = msk(np.nanmean(np.load(cnn_p), 0)) if os.path.exists(cnn_p) else gc_r
    vmax  = max(float(np.nanpercentile(gc_r[INDIA], 98)), 10.0)

    cl, cr = st.columns(2)
    with cl:
        t = f"GraphCast Forecast — {MONTH_NAMES[mo-1]} {yr}"
        if "Globe" in vtype:
            fig = globe({"GraphCast": gc_r}, vmax, t)
        elif "2D" in vtype:
            fig = map2d_plotly({"GraphCast": gc_r}, vmax, t)
        else:
            fig = surface3d(gc_r, vmax, t)
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))
    with cr:
        t = f"CNN Bias-Corrected — {MONTH_NAMES[mo-1]} {yr}"
        if "Globe" in vtype:
            fig = globe({"CNN Corrected": cnn_r}, vmax, t)
        elif "2D" in vtype:
            fig = map2d_plotly({"CNN Corrected": cnn_r}, vmax, t)
        else:
            fig = surface3d(cnn_r, vmax, t)
        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    st.markdown("""
    <div style="background:#040f1c;border:1px solid #1a3a5c;border-left:4px solid #00c8ff;
                border-radius:8px;padding:0.8rem 1.2rem;color:#5a8aaa;font-size:0.82rem;
                margin-top:0.5rem;">
      💡 <b style="color:#00c8ff;">Tip:</b> Drag to rotate the globe &nbsp;·&nbsp;
      Scroll to zoom &nbsp;·&nbsp; Double-click to reset view
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1 — RAINFALL MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif "Rainfall" in mode:
    st.markdown("""<div style="color:#00c8ff;font-size:1.3rem;font-weight:700;
    border-left:4px solid #00c8ff;padding-left:1rem;margin-bottom:1rem;">
    📊 Monthly Rainfall Monitor</div>""", unsafe_allow_html=True)

    ca, cb_col = st.columns([3,1])
    with ca: sel = st.selectbox("Month", avail_months, index=len(avail_months)-1)
    with cb_col: view = st.radio("View",["🌍 Globe","🗺️ 2D Map","⛰️ 3D Sfc"],
                                 label_visibility="collapsed")
    yr, mo = int(sel[:4]), int(sel[5:])

    gc_r  = msk(np.nanmean(np.load(f"{GC_RAIN}/{sel}.npy"), 0))
    cnn_r = msk(np.nanmean(np.load(f"{CNN_RAIN}/{sel}.npy"), 0)) \
            if os.path.exists(f"{CNN_RAIN}/{sel}.npy") else None
    imd_r = msk(np.nanmean(np.load(f"{IMD_RAIN}/{sel}.npy"), 0)) \
            if os.path.exists(f"{IMD_RAIN}/{sel}.npy") else None

    panels = {"GraphCast Raw": gc_r}
    if cnn_r is not None: panels["CNN Corrected"] = cnn_r
    if imd_r is not None: panels["IMD Observed"]  = imd_r
    vmax = max(float(np.nanpercentile(gc_r[INDIA], 98)), 10.0)

    cols = st.columns(len(panels))
    for col,(name,data) in zip(cols, panels.items()):
        with col:
            t = f"{name} — {MONTH_NAMES[mo-1]} {yr}"
            if "Globe" in view:
                st.plotly_chart(globe({name:data}, vmax, t),
                                use_container_width=True, config=dict(displayModeBar=False))
            elif "3D" in view:
                st.plotly_chart(surface3d(data, vmax, t),
                                use_container_width=True, config=dict(displayModeBar=False))
            else:
                fig = map2d(data, vmax, t)
                st.pyplot(fig, use_container_width=True); plt.close('all')

    if cnn_r is not None and imd_r is not None:
        mask_v = np.isfinite(imd_r)&np.isfinite(gc_r)&np.isfinite(cnn_r)
        o,g,c = imd_r[mask_v], gc_r[mask_v], cnn_r[mask_v]
        if len(o):
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("""<div style="color:#00c8ff;font-weight:600;
            border-left:3px solid #00c8ff;padding-left:0.8rem;">
            📈 Bias Correction Performance</div>""", unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            gm=float(np.mean(np.abs(o-g))); cm=float(np.mean(np.abs(o-c)))
            m1.metric("GC MAE",   f"{gm:.2f} mm/day")
            m2.metric("CNN MAE",  f"{cm:.2f} mm/day", delta=f"{gm-cm:+.2f}")
            m3.metric("GC Bias",  f"{float(np.mean(g-o)):+.2f}")
            m4.metric("CNN Bias", f"{float(np.mean(c-o)):+.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2 — TEMPERATURE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif "Temperature" in mode:
    st.markdown("""<div style="color:#ff8844;font-size:1.3rem;font-weight:700;
    border-left:4px solid #ff8844;padding-left:1rem;margin-bottom:1rem;">
    🌡️ Monthly Temperature Monitor</div>""", unsafe_allow_html=True)

    avail_t = sorted([os.path.basename(f).replace('.npy','')
                      for f in glob.glob(f"{GC_TEMP}/*.npy")])
    if not avail_t: st.warning("No temperature files found."); st.stop()

    ca, cb_col = st.columns([3,1])
    with ca: sel = st.selectbox("Month", avail_t, index=len(avail_t)-1)
    with cb_col: view = st.radio("View",["🌍 Globe","🗺️ 2D Map","⛰️ 3D Sfc"],
                                 label_visibility="collapsed")
    yr, mo = int(sel[:4]), int(sel[5:])

    def load_t(p):
        r=np.load(p); t=r if r.ndim==3 else r[:,:,:,0]
        return msk(np.nanmean(t,0))-273.15

    gc_t  = load_t(f"{GC_TEMP}/{sel}.npy")
    cnn_t = load_t(f"{CNN_TEMP}/{sel}.npy") if os.path.exists(f"{CNN_TEMP}/{sel}.npy") else None
    imd_t = load_t(f"{IMD_TEMP}/{sel}.npy") if os.path.exists(f"{IMD_TEMP}/{sel}.npy") else None

    panels = {"GraphCast": gc_t}
    if cnn_t is not None: panels["CNN Corrected"] = cnn_t
    if imd_t is not None: panels["IMD Observed"]  = imd_t

    all_v = np.concatenate([d[INDIA][np.isfinite(d[INDIA])] for d in panels.values()])
    vmin_t,vmax_t = float(np.percentile(all_v,2)),float(np.percentile(all_v,98))

    cols = st.columns(len(panels))
    for col,(name,data) in zip(cols, panels.items()):
        with col:
            t = f"{name} — {MONTH_NAMES[mo-1]} {yr}"
            if "Globe" in view:
                st.plotly_chart(globe({name:data},vmax_t,t,"°C",TEMP_CS,vmin_t),
                                use_container_width=True, config=dict(displayModeBar=False))
            elif "2D" in view:
                st.plotly_chart(map2d_plotly({name:data},vmax_t,t,"°C",TEMP_CS,vmin_t),
                                use_container_width=True, config=dict(displayModeBar=False))
            else:
                st.plotly_chart(surface3d(data,vmax_t,t,"°C",TEMP_CS,vmin_t),
                                use_container_width=True, config=dict(displayModeBar=False))

    if cnn_t is not None and imd_t is not None:
        mask_v = np.isfinite(imd_t)&np.isfinite(gc_t)&np.isfinite(cnn_t)
        o,g,c = imd_t[mask_v],gc_t[mask_v],cnn_t[mask_v]
        if len(o):
            st.markdown("<hr>", unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            gm=float(np.mean(np.abs(o-g))); cm=float(np.mean(np.abs(o-c)))
            m1.metric("GC MAE",   f"{gm:.2f} °C")
            m2.metric("CNN MAE",  f"{cm:.2f} °C", delta=f"{gm-cm:+.2f}")
            m3.metric("GC Bias",  f"{float(np.mean(g-o)):+.2f}")
            m4.metric("CNN Bias", f"{float(np.mean(c-o)):+.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3 — FLOOD EVENT REPLAY
# ══════════════════════════════════════════════════════════════════════════════
elif "Flood" in mode:
    st.markdown("""<div style="color:#ff4444;font-size:1.3rem;font-weight:700;
    border-left:4px solid #ff4444;padding-left:1rem;margin-bottom:1rem;">
    🌧️ Extreme Flood Event Replay</div>""", unsafe_allow_html=True)

    event = st.selectbox("Select Event", [
        "🔴  Kerala Floods — 15 August 2018",
        "🔴  Uttarakhand Flash Floods — 16 June 2013",
    ])

    if "Kerala" in event:
        month,day_idx,skw = "2018_08",14,"KERALA"
        etitle = "Kerala Floods — 15 August 2018"
    else:
        month,day_idx,skw = "2013_06",15,"UTTARAKHAND"
        etitle = "Uttarakhand Flash Floods — 16 June 2013"

    if not os.path.exists(f"{GC_RAIN}/{month}.npy"):
        st.error(f"File not found: {GC_RAIN}/{month}.npy"); st.stop()

    def _load_day(path, idx):
        arr = np.load(path).astype(float)
        if arr.ndim == 2:          # already (lat, lon) — monthly mean
            return arr
        if arr.ndim == 3:          # (time, lat, lon)
            return arr[min(idx, arr.shape[0]-1)]
        if arr.ndim == 4:          # (time, lev, lat, lon)
            return arr[min(idx, arr.shape[0]-1), 0]
        return arr[0] if arr.ndim > 2 else arr

    gc_d  = msk(_load_day(f"{GC_RAIN}/{month}.npy", day_idx))
    cnn_d = msk(_load_day(f"{CNN_RAIN}/{month}.npy", day_idx)) \
            if os.path.exists(f"{CNN_RAIN}/{month}.npy") else None
    imd_d = msk(_load_day(f"{IMD_RAIN}/{month}.npy", day_idx)) \
            if os.path.exists(f"{IMD_RAIN}/{month}.npy") else None

    _reg  = _gdf[_gdf['STATE'].str.contains(skw, case=False, na=False)]
    _rp   = prepared.prep(unary_union(_reg.geometry))
    pts   = [Point(lo,la) for la,lo in zip(LAT_2D.ravel(),LON_2D.ravel())]
    RMASK = np.array([_rp.contains(p) for p in pts]).reshape(LAT_2D.shape)

    view = st.radio("View Type",["🌍 Globe","🗺️ 2D Map","⛰️ 3D Surface"], horizontal=True)

    panels = {"GraphCast Raw": gc_d}
    if cnn_d is not None: panels["CNN Corrected"] = cnn_d
    if imd_d is not None: panels["IMD Observed"]  = imd_d

    vmax = max(float(np.nanpercentile(
        np.concatenate([d[INDIA][np.isfinite(d[INDIA])] for d in panels.values()]),99)),50.0)

    cols = st.columns(len(panels))
    for col,(name,data) in zip(cols, panels.items()):
        with col:
            t = f"{name}<br>{etitle}" if "Globe" in view or "3D" in view \
                else f"{name}\n{etitle}"
            if "Globe" in view:
                st.plotly_chart(globe({name:data},vmax,t),
                                use_container_width=True, config=dict(displayModeBar=False))
            elif "3D" in view:
                st.plotly_chart(surface3d(data,vmax,t),
                                use_container_width=True, config=dict(displayModeBar=False))
            else:
                fig=map2d(data,vmax,t,gdf_reg=_reg)
                st.pyplot(fig,use_container_width=True); plt.close('all')

    st.markdown("<hr>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("GC Mean — Region",  f"{np.nanmean(gc_d[RMASK]):.1f} mm/day")
    c2.metric("GC Max — Region",   f"{np.nanmax(gc_d[RMASK]):.1f} mm/day")
    if cnn_d is not None:
        c3.metric("CNN Mean — Region", f"{np.nanmean(cnn_d[RMASK]):.1f} mm/day")
    if imd_d is not None:
        c4.metric("IMD Mean — Region", f"{np.nanmean(imd_d[RMASK]):.1f} mm/day")

# ══════════════════════════════════════════════════════════════════════════════
# 4 — WHAT-IF SCENARIO
# ══════════════════════════════════════════════════════════════════════════════
elif "What-If" in mode:
    st.markdown("""<div style="color:#00e676;font-size:1.3rem;font-weight:700;
    border-left:4px solid #00e676;padding-left:1rem;margin-bottom:1rem;">
    🔮 What-If Climate Scenario Simulator</div>""", unsafe_allow_html=True)

    avail_c = sorted([os.path.basename(f).replace('.npy','')
                      for f in glob.glob(f"{CNN_RAIN}/*.npy")])
    if not avail_c: st.warning("No CNN corrected files found."); st.stop()

    c1,c2,c3 = st.columns([2,1,1])
    with c1: sel = st.selectbox("Base Month", avail_c, index=len(avail_c)-1)
    with c2: rain_pct = st.slider("Rainfall (%)", -60, +150, 0, step=5)
    with c3: temp_dC  = st.slider("Temperature (°C)", -3.0, +5.0, 0.0, step=0.5)

    yr, mo = int(sel[:4]), int(sel[5:])
    base   = msk(np.nanmean(np.load(f"{CNN_RAIN}/{sel}.npy"), 0))
    scen   = np.where(np.isfinite(base), base*(1+rain_pct/100), np.nan)
    vmax   = max(float(np.nanpercentile(
        np.concatenate([base[INDIA][np.isfinite(base[INDIA])],
                        scen[INDIA][np.isfinite(scen[INDIA])]]),98)),10.0)

    view = st.radio("View Type",["🌍 Globe","🗺️ 2D Map","⛰️ 3D Surface"], horizontal=True)
    cl,cr = st.columns(2)
    with cl:
        t=f"Current State — {MONTH_NAMES[mo-1]} {yr}"
        if "Globe" in view:   fig = globe({"Current":base},vmax,t)
        elif "2D" in view:    fig = map2d_plotly({"Current":base},vmax,t)
        else:                  fig = surface3d(base,vmax,t)
        st.plotly_chart(fig,use_container_width=True,config=dict(displayModeBar=False))
    with cr:
        t=f"Scenario: {rain_pct:+d}%  |  {temp_dC:+.1f}°C"
        if "Globe" in view:   fig = globe({"Scenario":scen},vmax,t)
        elif "2D" in view:    fig = map2d_plotly({"Scenario":scen},vmax,t)
        else:                  fig = surface3d(scen,vmax,t)
        st.plotly_chart(fig,use_container_width=True,config=dict(displayModeBar=False))

    st.markdown("<hr>", unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    bm=float(np.nanmean(base[INDIA])); sm=float(np.nanmean(scen[INDIA]))
    m1.metric("Baseline Mean",   f"{bm:.1f} mm/day")
    m2.metric("Scenario Mean",   f"{sm:.1f} mm/day", delta=f"{sm-bm:+.1f}")
    m3.metric("Rainfall Change", f"{rain_pct:+d}%")
    m4.metric("Temp Shift",      f"{temp_dC:+.1f} °C")

    if rain_pct > 40:
        st.markdown("""<div style="background:#1a0000;border-left:4px solid #ff4444;
        border-radius:8px;padding:0.8rem 1.2rem;color:#ff8888;">
        🚨 <b>High flood risk scenario</b></div>""", unsafe_allow_html=True)
    elif rain_pct > 20:
        st.markdown("""<div style="background:#1a0e00;border-left:4px solid #ff9800;
        border-radius:8px;padding:0.8rem 1.2rem;color:#ffcc80;">
        ⚠️ <b>Elevated flood risk</b> — above-normal rainfall</div>""", unsafe_allow_html=True)
    elif rain_pct < -25:
        st.markdown("""<div style="background:#1a1200;border-left:4px solid #ff9800;
        border-radius:8px;padding:0.8rem 1.2rem;color:#ffcc80;">
        ⚠️ <b>Drought risk</b> — below-normal rainfall</div>""", unsafe_allow_html=True)
    elif temp_dC >= 2.0:
        st.markdown(f"""<div style="background:#1a0800;border-left:4px solid #ff6600;
        border-radius:8px;padding:0.8rem 1.2rem;color:#ffaa60;">
        ⚠️ <b>+{temp_dC}°C warming</b> — heatwave risk elevated</div>""",
        unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:#001a0a;border-left:4px solid #00c853;
        border-radius:8px;padding:0.8rem 1.2rem;color:#80e8a0;">
        ✅ <b>Normal range</b> — within historical climate bounds</div>""",
        unsafe_allow_html=True)
