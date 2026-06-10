"""
PPE Compliance — Dashboard
============================
Pre-computed results + Live GPU inference via Modal T4.
"""

import streamlit as st
import json, base64, requests
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="PPE Compliance Monitor", page_icon="🦺", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .hero { background: linear-gradient(135deg, #e8f0fe 0%, #e6f4ea 100%); border-radius: 16px; padding: 28px 32px; margin-bottom: 24px; border: 1px solid #d2e3fc; }
    .hero h1 { color: #1a1a2e; font-size: 2rem; font-weight: 700; margin: 0; }
    .hero p { color: #5f6368; font-size: 0.95rem; margin: 6px 0 0; }
    .badge { display:inline-block; padding:3px 10px; border-radius:10px; font-size:0.75rem; font-weight:600; margin-right:4px; }
    .badge-base { background:#e8f0fe; color:#1a73e8; }
    .badge-sam { background:#e6f4ea; color:#137333; }
    .card { background:#f8f9fa; border-radius:12px; padding:20px; border:1px solid #e8eaed; }
</style>
""", unsafe_allow_html=True)

MODAL_URL = "https://sshivamvyas--ppe-compliance-detect-http.modal.run"

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    results = {}
    for p in sorted(Path("outputs").glob("*.json")):
        with open(p) as f: data = json.load(f)
        model = "sam" if "sam" in p.stem.lower() else "baseline"
        vname = p.stem.replace("input_","").replace("_baseline","").replace("_sam","")
        results.setdefault(vname, {})[model] = data
    return results

# ═══════════════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <h1>🦺 PPE Compliance Tracking System</h1>
    <p>
        Comparing <span class="badge badge-base">Baseline YOLO11m</span> (trained directly on Construction-PPE)
        vs <span class="badge badge-sam">SAM-Teacher YOLO11m</span> (9,994 SAM pseudo-labels) — 55% mAP improvement
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 📈 Models")
st.sidebar.markdown('<span class="badge badge-base">Baseline</span> mAP50: **0.558**', unsafe_allow_html=True)
st.sidebar.markdown('<span class="badge badge-sam">SAM-Teacher</span> mAP50: **0.864**', unsafe_allow_html=True)
st.sidebar.markdown("**+55%** with 9,994 SAM pseudo-labels")
st.sidebar.markdown("---")

# ── Live GPU ─────────────────────────────────────────────────────────────────

st.markdown("### ⚡ Upload & Detect on GPU")

video_file = st.file_uploader("Upload video for live GPU processing", type=["mp4","avi","mov","webm"],
                               help="Both Baseline & SAM-Teacher models run simultaneously on a Modal T4 GPU")

if video_file:
    if "last_video" not in st.session_state or st.session_state.last_video != video_file.name:
        st.session_state.last_video = video_file.name
        st.session_state.gpu_result = None

    st.video(video_file)
    st.caption(f"{video_file.name} — {video_file.size/1e6:.1f} MB")

    if st.button("🚀 Process on T4 GPU", type="primary", use_container_width=True):
        with st.spinner("⚡ Running both models on T4 GPU..."):
            try:
                video_bytes = video_file.read()
                resp = requests.post(
                    MODAL_URL,
                    json={"video_b64": base64.b64encode(video_bytes).decode(), "video_name": video_file.name},
                    timeout=600,
                )
                if resp.status_code != 200:
                    st.error(f"Modal error: {resp.status_code}")
                else:
                    result = resp.json()
                    if "error" in result:
                        st.error(f"Modal: {result['error'][:500]}")
                    else:
                        st.session_state.gpu_result = result
                        st.success(f"✅ {result['results']['baseline']['frames']} frames processed on T4")
                        st.balloons()
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # Show GPU results
    if "gpu_result" in st.session_state and st.session_state.gpu_result:
        r = st.session_state.gpu_result["results"]
        rb, rs = r["baseline"], r["sam"]

        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline", f"{rb['total_detections']:,}", f"{rb['total_detections']/max(rb['frames'],1):.1f}/frame")
        c2.metric("SAM-Teacher", f"{rs['total_detections']:,}", f"{rs['total_detections']/max(rs['frames'],1):.1f}/frame")
        c3.metric("Δ SAM vs Baseline", f"{rs['total_detections']-rb['total_detections']:+,}")

        # Bar chart
        all_c = sorted(set(list(rb['class_totals'])+list(rs['class_totals'])))
        df = pd.DataFrame([{"Class":c,"Baseline":rb['class_totals'].get(c,0),"SAM-Teacher":rs['class_totals'].get(c,0)} for c in all_c])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Class"],y=df["Baseline"],name="Baseline",marker_color="#1a73e8",text=df["Baseline"],textposition="outside"))
        fig.add_trace(go.Bar(x=df["Class"],y=df["SAM-Teacher"],name="SAM-Teacher",marker_color="#137333",text=df["SAM-Teacher"],textposition="outside"))
        fig.update_layout(height=400,barmode="group",margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)

        # Side-by-side preview images
        st.markdown("### 📸 Annotated Preview (first frame)")
        cv1, cv2 = st.columns(2)
        with cv1:
            st.markdown('<span class="badge badge-base">Baseline</span>', unsafe_allow_html=True)
            if rb.get("preview_b64"):
                st.image(base64.b64decode(rb["preview_b64"]), use_container_width=True)
        with cv2:
            st.markdown('<span class="badge badge-sam">SAM-Teacher</span>', unsafe_allow_html=True)
            if rs.get("preview_b64"):
                st.image(base64.b64decode(rs["preview_b64"]))

        # Download buttons
        st.markdown("### 📥 Download Annotated Videos (first 15 sec)")
        dc1, dc2 = st.columns(2)
        with dc1:
            if rb.get("video_b64"):
                st.download_button("⬇ Download Baseline Video", base64.b64decode(rb["video_b64"]),
                                   f"baseline_annotated.mp4", "video/mp4", use_container_width=True)
            else:
                st.caption("Baseline video unavailable")
        with dc2:
            if rs.get("video_b64"):
                st.download_button("⬇ Download SAM-Teacher Video", base64.b64decode(rs["video_b64"]),
                                   "SAM_teacher_annotated.mp4", "video/mp4", use_container_width=True)
            else:
                st.caption("SAM video unavailable")

# ── Pre-computed ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 📊 Pre-computed Results")

all_results = load_results()
if all_results:
    selected = st.sidebar.selectbox("📹 Pre-computed video", sorted(all_results.keys()),
                                     help="Switch between previously processed videos")
    data = all_results[selected]
    base, sam = data.get("baseline"), data.get("sam")
    if base:
        st.markdown(f"**{selected}** — Baseline: {base['total_detections']:,} detections"
                    + (f", SAM: {sam['total_detections']:,}" if sam else " (SAM data pending)"))
