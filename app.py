"""
PPE Compliance — Dashboard (Zero-GPU)
=======================================
Loads pre-processed detection results from both models.
Shows side-by-side comparison, charts, and annotated videos.

Deploy anywhere: HF Spaces, Streamlit Cloud, local CPU.

Usage:
  streamlit run app.py
"""

import streamlit as st
import json
import glob
from pathlib import Path
from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="PPE Compliance Monitor", page_icon="🦺", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin: 0; }
    .sub-header { color: #666; font-size: 0.95rem; margin-bottom: 1rem; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 10px;
             font-size: 0.75rem; font-weight: 600; }
    .badge-base { background: #e8f0fe; color: #1a73e8; }
    .badge-sam { background: #e6f4ea; color: #137333; }
    .metric-card { text-align: center; padding: 12px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; color: #666; }
    .winner { color: #137333; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    """Load all pre-processed detection JSON files."""
    results = {"baseline": {}, "sam": {}}
    for p in sorted(Path("outputs").glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        model = "sam" if "sam" in p.stem.lower() else "baseline"
        video_name = p.stem.replace(f"_{model}", "").replace(f"_{model.capitalize()}", "")
        results[model][video_name] = data
    return results

# ── UI ───────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🦺 PPE Compliance Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">SAM-Teacher Pipeline — Baseline vs SAM-Refined Comparison</p>', unsafe_allow_html=True)

results = load_results()

if not results["baseline"] and not results["sam"]:
    st.warning("""
    No pre-processed results found. Run locally first:
    ```
    python preprocess.py --video ../ppe-compliance/data/video/input.mp4
    ```
    Then place the output files in `outputs/`.
    """)

    # Still show model comparison stats
    st.markdown("### 📈 Model Training Results (Test Set)")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Baseline mAP50", "0.558", delta="11-class YOLO11m")
    with c2:
        st.metric("SAM-Refined mAP50", "0.864", delta="+55.0%", delta_color="normal")

    st.image("https://via.placeholder.com/800x400/e8f0fe/1a73e8?text=PPE+Compliance+Pipeline", use_container_width=True)
    st.stop()

# ── Video Selector ───────────────────────────────────────────────────────────
all_videos = sorted(set(list(results["baseline"].keys()) + list(results["sam"].keys())))

st.sidebar.header("📹 Select Video")
selected_video = st.sidebar.selectbox("Video", all_videos) if all_videos else None

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Stats")
st.sidebar.markdown('<span class="badge badge-base">Baseline</span> mAP50: **0.558** • 11 classes', unsafe_allow_html=True)
st.sidebar.markdown('<span class="badge badge-sam">SAM-Refined</span> mAP50: **0.864** • 6 classes', unsafe_allow_html=True)
st.sidebar.markdown(f"**Improvement:** +55.0% mAP50, +152.7% mAP50-95")

if not selected_video:
    st.stop()

base_data = results["baseline"].get(selected_video)
sam_data = results["sam"].get(selected_video)

# ── Summary Metrics ──────────────────────────────────────────────────────────
st.markdown(f"### 🎯 Results: `{selected_video}`")

col_b, col_s = st.columns(2)

with col_b:
    st.markdown(f'<span class="badge badge-base">Baseline</span>', unsafe_allow_html=True)
    if base_data:
        b_total = base_data["total_detections"]
        b_frames = base_data["frames"]
        b_avg = b_total / max(b_frames, 1)
        st.metric("Total Detections", f"{b_total:,}", f"{b_avg:.1f}/frame")
    else:
        st.info("Not processed")

with col_s:
    st.markdown(f'<span class="badge badge-sam">SAM-Refined</span>', unsafe_allow_html=True)
    if sam_data:
        s_total = sam_data["total_detections"]
        s_frames = sam_data["frames"]
        s_avg = s_total / max(s_frames, 1)
        st.metric("Total Detections", f"{s_total:,}", f"{s_avg:.1f}/frame")
    else:
        st.info("Not processed")

# ── Per-Class Comparison ────────────────────────────────────────────────────
st.markdown("### 📊 Per-Class Detection Breakdown")

if base_data and sam_data:
    all_classes = sorted(set(list(base_data.get("class_totals", {}).keys()) +
                             list(sam_data.get("class_totals", {}).keys())))
    class_comparison = []
    for cls in all_classes:
        b_c = base_data.get("class_totals", {}).get(cls, 0)
        s_c = sam_data.get("class_totals", {}).get(cls, 0)
        class_comparison.append({"Class": cls, "Baseline": b_c, "SAM-Refined": s_c})

    df = pd.DataFrame(class_comparison)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Class"], y=df["Baseline"], name="Baseline",
                         marker_color="#1a73e8", text=df["Baseline"], textposition="outside"))
    fig.add_trace(go.Bar(x=df["Class"], y=df["SAM-Refined"], name="SAM-Refined",
                         marker_color="#137333", text=df["SAM-Refined"], textposition="outside"))
    fig.update_layout(height=400, barmode="group", margin=dict(l=0,r=0,t=10,b=0),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Class Data Table"):
        st.dataframe(df, hide_index=True, use_container_width=True)

# ── Side-by-Side Videos ─────────────────────────────────────────────────────
st.markdown("### 🎬 Side-by-Side Comparison")

col_v1, col_v2 = st.columns(2)

with col_v1:
    st.markdown(f'<span class="badge badge-base">Baseline (mAP50=0.558)</span>', unsafe_allow_html=True)
    base_video = f"outputs/{selected_video}_baseline.mp4"
    if Path(base_video).exists():
        st.video(base_video)
    else:
        st.info("Annotated video not found")

with col_v2:
    st.markdown(f'<span class="badge badge-sam">SAM-Refined (mAP50=0.864)</span>', unsafe_allow_html=True)
    sam_video = f"outputs/{selected_video}_sam.mp4"
    if Path(sam_video).exists():
        st.video(sam_video)
    else:
        st.info("Annotated video not found")

# ── Training Metrics ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Model Training Comparison")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("mAP50", "0.558→0.864", "+55.0%")
with col_m2:
    st.metric("mAP50-95", "0.281→0.710", "+152.7%")
with col_m3:
    st.metric("Precision", "0.674→0.870", "+29.1%")
with col_m4:
    st.metric("Recall", "0.612→0.804", "+31.4%")

st.markdown("### 🔧 SAM-Teacher Pipeline")
st.code("""
Raw Images → Grounding DINO → SAM 1 (Masks) → Tight YOLO BBoxes → YOLO11m Student
  (1,416)     (text→boxes)     (box→mask)      (mask→bbox)        (100 epochs)
                                    ↓
                          mAP50: 0.558 → 0.864 (+55%)
                       mAP50-95: 0.281 → 0.710 (+153%)
""")

# ── Live GPU Inference (Modal) ───────────────────────────────────────────────

st.markdown("---")
st.markdown("### ⚡ Live GPU Inference (via Modal)")

st.markdown("""
Upload any video and process it on a **T4 GPU** — results in seconds instead of minutes.
Requires Modal to be set up (see setup instructions below).
""")

with st.expander("🔧 Modal Setup (one-time)", expanded=False):
    st.markdown("""
    **1. Install Modal:**
    ```bash
    pip install modal
    modal setup    # logs into your Modal account
    ```

    **2. Upload models to Modal Volume:**
    ```bash
    cd ppe-deploy-phase
    python -c "import modal; from modal_inference import upload_model, app; modal.runner.deploy_stub(app)"
    modal run modal_inference.py::upload_model
    ```

    **3. Deploy the inference function:**
    ```bash
    modal deploy modal_inference.py
    ```
    """)

st.info(
    "🎯 Once Modal is deployed, upload any video here and it processes on a free T4 GPU "
    "in ~10 seconds. You get $30/month free credits — enough for thousands of inferences."
)

# Add a "coming soon" uploader that will work once Modal is deployed
demo_video = st.file_uploader(
    "Upload video for GPU inference (after Modal setup)",
    type=["mp4", "avi", "mov", "webm"],
    disabled=True,  # Enable after Modal deployment
    help="Upload gets enabled after Modal deployment"
)

if demo_video:
    st.video(demo_video)
    if st.button("🚀 Process on GPU (Modal)", type="primary"):
        with st.spinner("Sending to Modal GPU..."):
            st.write("Processing... (requires Modal deployment)")
            # Uncomment after Modal setup:
            # import modal
            # f = modal.Function.lookup("ppe-compliance", "detect_video")
            # result = f.remote(video_url=uploaded_url, model="sam")
            # ... render results
else:
    st.caption("Upload will be available after Modal is deployed and configured.")

