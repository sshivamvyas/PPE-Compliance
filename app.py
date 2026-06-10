"""
PPE Compliance — Dashboard
============================
Shows pre-computed detection results from Baseline YOLO11m vs SAM-Teacher YOLO11m.
Side-by-side video comparison, per-class charts, and detailed metrics.

Add your own videos: process locally with run_video.py → push JSONs → refresh.

Usage: streamlit run app.py
"""

import streamlit as st
import json
import glob
from pathlib import Path
from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="PPE Compliance Monitor", page_icon="🦺", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin: 0; }
    .sub-header { color: #666; font-size: 0.95rem; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }
    .badge-base { background: #e8f0fe; color: #1a73e8; }
    .badge-sam { background: #e6f4ea; color: #137333; }
    .metric-box { background: #f8f9fa; border-radius: 12px; padding: 16px; text-align: center; }
    .metric-box h2 { margin: 0; font-size: 2rem; }
    .metric-box p { margin: 4px 0 0; color: #666; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Modal URL (replace after `modal deploy`) ─────────────────────────────────
MODAL_URL = "https://sshivamvyas--ppe-compliance-detect-http.modal.run"

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    results = {}
    for p in sorted(Path("outputs").glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        model = "sam" if "sam" in p.stem.lower() else "baseline"
        vname = p.stem.replace("input_", "").replace("_baseline", "").replace("_sam", "").replace("_Baseline", "").replace("_Sam", "")
        if vname not in results:
            results[vname] = {}
        results[vname][model] = data
    return results

def find_video(video_name, model):
    """Find preview video if available."""
    patterns = [
        f"outputs/input_{video_name}_{model}_preview.mp4",
        f"outputs/input_{video_name}_preview.mp4",
        f"outputs/input_{video_name}_{model}.mp4",
    ]
    for pat in patterns:
        if Path(pat).exists():
            return pat
    return None

# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">🦺 PPE Compliance Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Baseline YOLO11m vs SAM-Teacher YOLO11m — Live Comparison Dashboard</p>', unsafe_allow_html=True)

all_results = load_results()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 📈 Model Comparison")
st.sidebar.markdown('<span class="badge badge-base">Baseline</span> mAP50: **0.558**', unsafe_allow_html=True)
st.sidebar.markdown('<span class="badge badge-sam">SAM-Teacher</span> mAP50: **0.864**', unsafe_allow_html=True)
st.sidebar.markdown("**+55% improvement** — with 9,994 SAM pseudo-labels")

st.sidebar.markdown("---")

if not all_results:
    st.sidebar.warning("No results found in outputs/")
    video_options = []
else:
    video_options = sorted(all_results.keys())

if video_options:
    selected = st.sidebar.selectbox("📹 Select Video", video_options)
else:
    selected = None
    st.warning("""
    ### No pre-computed results yet

    Process a video locally to populate this dashboard:
    ```bash
    cd ppe-deploy-phase
    python preprocess.py --video your_video.mp4
    ```

    Then push the generated JSON files to GitHub. The dashboard updates automatically.
    """)

if not selected:
    st.info("No videos available. See sidebar for how to add one.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
#  Results for selected video
# ═══════════════════════════════════════════════════════════════════════════════

data = all_results[selected]
base = data.get("baseline")
sam = data.get("sam")

st.markdown(f"### 🎯 Video: `{selected}`")

# ── KPI Cards ────────────────────────────────────────────────────────────────
cols = st.columns(4)
with cols[0]:
    st.metric(
        "Baseline Detections",
        f"{base['total_detections']:,}" if base else "N/A",
        delta=f"{base['total_detections']/max(base['frames'],1):.1f}/frame" if base else None,
    )
with cols[1]:
    st.metric(
        "SAM-Teacher Detections",
        f"{sam['total_detections']:,}" if sam else "N/A",
        delta=f"{sam['total_detections']/max(sam['frames'],1):.1f}/frame" if sam else None,
    )
with cols[2]:
    val = base['total_detections'] if base else 0
    st.metric("Frames Processed", f"{base['frames']}" if base else "N/A")
with cols[3]:
    if base and sam:
        improvement = ((sam['total_detections'] - base['total_detections']) / max(base['total_detections'], 1)) * 100
        st.metric("SAM vs Baseline", f"{improvement:+.1f}%", "more detections")

# ── Per-Class Comparison Chart ───────────────────────────────────────────────
if base and sam:
    st.markdown("### 📊 Per-Class Detections")

    all_classes = sorted(set(
        list(base.get("class_totals", {}).keys()) +
        list(sam.get("class_totals", {}).keys())
    ))

    df = pd.DataFrame([
        {
            "Class": c,
            "Baseline": base["class_totals"].get(c, 0),
            "SAM-Teacher": sam["class_totals"].get(c, 0),
        }
        for c in all_classes
    ])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Class"], y=df["Baseline"], name="Baseline",
                         marker_color="#1a73e8", text=df["Baseline"], textposition="outside"))
    fig.add_trace(go.Bar(x=df["Class"], y=df["SAM-Teacher"], name="SAM-Teacher",
                         marker_color="#137333", text=df["SAM-Teacher"], textposition="outside"))
    fig.update_layout(
        height=400, barmode="group",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    with st.expander("📋 Data Table"):
        df_disp = df.copy()
        if "Baseline" in df_disp.columns and "SAM-Teacher" in df_disp.columns:
            df_disp["Δ"] = df_disp["SAM-Teacher"] - df_disp["Baseline"]
        st.dataframe(df_disp, hide_index=True, use_container_width=True)

# ── Side-by-Side Videos ──────────────────────────────────────────────────────
st.markdown("### 🎬 Annotated Videos")

video_base = find_video(selected, "baseline")
video_sam = find_video(selected, "sam")

col_v1, col_v2 = st.columns(2)

with col_v1:
    st.markdown('<span class="badge badge-base">Baseline YOLO11m</span> (mAP50: 0.558)', unsafe_allow_html=True)
    if video_base and Path(video_base).exists():
        st.video(video_base)
    else:
        st.warning("Preview video not found (too large for GitHub). Process locally to generate.")

with col_v2:
    st.markdown('<span class="badge badge-sam">SAM-Teacher YOLO11m</span> (mAP50: 0.864)', unsafe_allow_html=True)
    if video_sam and Path(video_sam).exists():
        st.video(video_sam)
    else:
        st.warning("Preview video not found (too large for GitHub). Process locally to generate.")

# ── Time-Series: Detections per frame ────────────────────────────────────────
if base and sam:
    st.markdown("### 📈 Detection Density Over Time")

    base_ts = [(r["time"], r["detections"]) for r in base.get("records", [])]
    sam_ts = [(r["time"], r["detections"]) for r in sam.get("records", [])]

    fig2 = go.Figure()
    if base_ts:
        t, d = zip(*base_ts)
        fig2.add_trace(go.Scatter(x=t, y=d, mode="lines", name="Baseline",
                                  line=dict(color="#1a73e8", width=1), opacity=0.7))
    if sam_ts:
        t, d = zip(*sam_ts)
        fig2.add_trace(go.Scatter(x=t, y=d, mode="lines", name="SAM-Teacher",
                                  line=dict(color="#137333", width=2), opacity=0.7))
    fig2.update_layout(height=350, xaxis_title="Time (seconds)", yaxis_title="Detections per frame",
                       margin=dict(l=0, r=0, t=10, b=0),
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, use_container_width=True)

# ── Process your own video ───────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Add Your Own Video")
st.sidebar.markdown("""
**Pre-computed:** Run locally with GPU:
```bash
python preprocess.py --video your_video.mp4
```
Push JSONs to GitHub → auto-updates.

**Live GPU:** Upload any video below for instant T4 GPU processing.
""")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE GPU Inference via Modal
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### ⚡ Live GPU Inference (Modal T4)")

# MODAL_URL = "https://sshivamvyas--ppe-compliance-detect-http.modal.run"
# After deploying Modal, replace the URL above with yours from `modal deploy` output.

with st.expander("🔧 Modal Setup (one-time)", expanded=False):
    st.markdown("""
    **1. Authenticate:**
    ```bash
    modal setup    # opens browser, sign up with GitHub
    ```

    **2. Upload models + deploy:**
    ```bash
    cd ppe-deploy-clean
    modal run modal_inference.py::upload_models
    modal deploy modal_inference.py
    ```

    **3. Copy the URL** printed after deploy (ends with `.modal.run`)
    and paste it into `MODAL_URL` in `app.py`. Then push to GitHub.
    """)

# ── Live GPU Inference (HTTP call to Modal) ──────────────────────────────────

st.markdown("### ⚡ Live GPU Inference (Modal T4)")

video_file = st.file_uploader(
    "Upload a video for GPU processing",
    type=["mp4", "avi", "mov", "webm"],
    help="Video is processed on Modal T4 GPU (~10-15 sec)",
    key="live_upload",
)

if video_file:
    st.video(video_file)
    st.caption(f"{video_file.name} — {video_file.size/1e6:.1f} MB")

    if st.button("🚀 Process on GPU", type="primary", use_container_width=True):
        with st.spinner("⚡ Sending to Modal T4 GPU... this takes ~10-15 seconds"):

            try:
                import requests
                import base64

                # Read video bytes once
                video_bytes = video_file.read()
                video_b64 = base64.b64encode(video_bytes).decode()

                response = requests.post(
                    MODAL_URL if MODAL_URL else "http://localhost:8000",
                    json={"video_b64": video_b64, "video_name": video_file.name},
                    timeout=300,
                )

                if response.status_code != 200:
                    st.error(f"Modal returned {response.status_code}: {response.text[:200]}")
                else:
                    result = response.json()
                    if "error" in result:
                        st.error(f"Modal error: {result['error']}")
                    else:
                        r_b = result["results"]["baseline"]
                        r_s = result["results"]["sam"]

                        st.success(f"✅ Done! {r_b['frames']} frames processed on T4")
                        st.balloons()

                        # KPIs
                        c1, c2 = st.columns(2)
                        c1.metric("Baseline", f"{r_b['total_detections']:,}",
                                  f"{r_b['total_detections']/max(r_b['frames'],1):.1f}/frame")
                        c2.metric("SAM-Teacher", f"{r_s['total_detections']:,}",
                                  f"{r_s['total_detections']/max(r_s['frames'],1):.1f}/frame")

                        # Charts
                        all_c = sorted(set(list(r_b['class_totals'].keys()) + list(r_s['class_totals'].keys())))
                        df = pd.DataFrame([{"Class":c,"Baseline":r_b['class_totals'].get(c,0),
                                           "SAM-Teacher":r_s['class_totals'].get(c,0)} for c in all_c])
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=df["Class"],y=df["Baseline"],name="Baseline",
                                             marker_color="#1a73e8",text=df["Baseline"],textposition="outside"))
                        fig.add_trace(go.Bar(x=df["Class"],y=df["SAM-Teacher"],name="SAM-Teacher",
                                             marker_color="#137333",text=df["SAM-Teacher"],textposition="outside"))
                        fig.update_layout(height=400,barmode="group",margin=dict(l=0,r=0,t=10,b=0))
                        st.plotly_chart(fig, use_container_width=True)

                        # Videos
                        st.markdown("### 🎬 Annotated Videos")
                        cv1, cv2 = st.columns(2)
                        with cv1:
                            st.markdown("**Baseline**")
                            st.video(base64.b64decode(r_b["video_b64"]))
                        with cv2:
                            st.markdown("**SAM-Teacher**")
                            st.video(base64.b64decode(r_s["video_b64"]))

            except Exception as e:
                st.warning(f"Modal not responding: {e}")
                st.info("Deploy Modal first (see expander above), then paste the URL.")

st.markdown("---")
st.markdown("""
<small>
**Approach:** Baseline YOLO11m trained directly on Construction-PPE →
SAM-Teacher (Grounding DINO + SAM 1) generates 9,994 pseudo-labels →
YOLO11m student on refined labels. mAP50: 0.558 → 0.864 (+55%).
</small>
""", unsafe_allow_html=True)
