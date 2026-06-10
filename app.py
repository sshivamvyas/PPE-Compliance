"""
PPE Compliance — Live Dashboard
==================================
Two modes:
  1. Pre-computed Results — loads saved JSONs (instant)
  2. Upload & Detect — upload any video, process on CPU, see results

Usage: streamlit run app.py
"""

import streamlit as st
import json
import cv2
import numpy as np
import tempfile
import os
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
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = {0:"helmet",1:"gloves",2:"vest",3:"boots",4:"goggles",6:"Person"}
SAM_REMAP = {0:6,1:0,2:2,3:1,4:4,5:3}
COLORS = {0:(255,255,0),1:(255,0,255),2:(0,165,255),3:(255,0,0),4:(0,255,255),6:(0,255,0)}

HF_MODELS = {
    "baseline": "sshivamvyas/ppe-compliance-baseline",
    "sam": "sshivamvyas/ppe-compliance-sam",
}

# ── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Download models from Hugging Face Hub (first time) or load from cache."""
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    import torch

    models = {}
    for name, repo in HF_MODELS.items():
        try:
            path = hf_hub_download(repo_id=repo, filename="best.pt")
            models[name] = YOLO(path)
        except Exception as e:
            st.warning(f"Could not load {name} model: {e}")
            models[name] = None
    return models

# ── Inference ────────────────────────────────────────────────────────────────
def detect_frame(model, frame, is_sam=False):
    results = model(frame, verbose=False)
    if results[0].boxes is None: return np.array([]), np.array([], dtype=int), np.array([])
    boxes = results[0].boxes.xyxy.cpu().numpy()
    cls = results[0].boxes.cls.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()
    if is_sam:
        for i in range(len(cls)): cls[i] = SAM_REMAP.get(int(cls[i]), int(cls[i]))
    return boxes, cls, confs

def draw_boxes(frame, boxes, cls_ids, confs):
    disp = frame.copy()
    for i in range(len(boxes)):
        x1,y1,x2,y2 = map(int, boxes[i])
        cid = int(cls_ids[i])
        color = COLORS.get(cid, (128,128,128))
        cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
        name = CLASS_NAMES.get(cid, f"c{cid}")
        cv2.putText(disp, f"{name} {confs[i]:.2f}", (x1,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return disp

def process_video(model, video_path, is_sam=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    records = []
    class_totals = {}
    total_dets = 0
    progress = st.progress(0)
    status = st.empty()

    for idx in range(total):
        ret, frame = cap.read()
        if not ret: break
        boxes, cls_ids, confs = detect_frame(model, frame, is_sam)

        if len(boxes) > 0:
            frame = draw_boxes(frame, boxes, cls_ids, confs)
        writer.write(frame)

        rec = {"frame":idx,"time":round(idx/fps,2),"detections":int(len(boxes)),"classes":{}}
        for cid in cls_ids:
            name = CLASS_NAMES.get(int(cid), f"c{int(cid)}")
            rec["classes"][name] = rec["classes"].get(name, 0) + 1
            class_totals[name] = class_totals.get(name, 0) + 1
        total_dets += len(boxes)
        records.append(rec)

        if idx % max(1, total//20) == 0:
            progress.progress(min(1.0, idx/max(1,total-1)))
            status.text(f"Frame {idx}/{total}")

    cap.release(); writer.release()
    progress.progress(1.0); status.empty()
    return records, out_path, class_totals, total_dets

# ── Load Pre-computed ────────────────────────────────────────────────────────
@st.cache_data
def load_precomputed():
    results = {"baseline": {}, "sam": {}}
    for p in sorted(Path("outputs").glob("*.json")):
        with open(p) as f: data = json.load(f)
        model = "sam" if "sam" in p.stem.lower() else "baseline"
        vname = p.stem.replace(f"_{model}","").replace(f"_{model.capitalize()}","")
        results[model][vname] = data
    return results

# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">🦺 PPE Compliance Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Baseline vs SAM-Teacher — Live Comparison</p>', unsafe_allow_html=True)

mode = st.sidebar.radio("Mode", ["📊 Pre-computed Results", "📤 Upload & Detect"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Model Stats")
st.sidebar.markdown('<span class="badge badge-base">Baseline</span> mAP50: **0.558**', unsafe_allow_html=True)
st.sidebar.markdown('<span class="badge badge-sam">SAM-Refined</span> mAP50: **0.864**', unsafe_allow_html=True)
st.sidebar.markdown("**+55% improvement**")

# ═══════════════════════════════════════════════════════════════════════════════
#  MODE 1: Pre-computed
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "📊 Pre-computed Results":
    results = load_precomputed()
    all_videos = sorted(set(list(results["baseline"].keys()) + list(results["sam"].keys())))

    if not all_videos:
        st.info("No pre-computed results. Switch to Upload & Detect mode.")
        st.stop()

    selected = st.sidebar.selectbox("Video", all_videos)
    base_data = results["baseline"].get(selected)
    sam_data = results["sam"].get(selected)

    st.markdown(f"### 🎯 Results: `{selected}`")

    col_b, col_s = st.columns(2)
    with col_b:
        st.markdown('<span class="badge badge-base">Baseline</span>', unsafe_allow_html=True)
        if base_data: st.metric("Detections", f"{base_data['total_detections']:,}", f"{base_data['total_detections']/max(base_data['frames'],1):.1f}/frame")
    with col_s:
        st.markdown('<span class="badge badge-sam">SAM-Refined</span>', unsafe_allow_html=True)
        if sam_data: st.metric("Detections", f"{sam_data['total_detections']:,}", f"{sam_data['total_detections']/max(sam_data['frames'],1):.1f}/frame")

    if base_data and sam_data:
        all_classes = sorted(set(list(base_data.get("class_totals",{}).keys()) + list(sam_data.get("class_totals",{}).keys())))
        df = pd.DataFrame([{"Class":c, "Baseline": base_data["class_totals"].get(c,0), "SAM": sam_data["class_totals"].get(c,0)} for c in all_classes])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Class"],y=df["Baseline"],name="Baseline",marker_color="#1a73e8"))
        fig.add_trace(go.Bar(x=df["Class"],y=df["SAM"],name="SAM-Refined",marker_color="#137333"))
        fig.update_layout(height=400, barmode="group", margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODE 2: Upload & Detect
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("### 📤 Upload Video for Live Detection")

    st.info("""
    **How it works:** Upload any video → both models process it on CPU → see results side by side.
    Processing speed: ~1-2 frames/sec on CPU (a 10-second video takes ~3-5 minutes).
    For faster processing, use the preprocessing script locally with a GPU.
    """)

    video_file = st.file_uploader("Choose a video file", type=["mp4","avi","mov","webm"])

    if video_file:
        # Save uploaded video
        tmp_video = tempfile.mktemp(suffix=f".{video_file.name.split('.')[-1]}")
        with open(tmp_video, "wb") as f:
            f.write(video_file.read())

        st.video(video_file)
        st.caption(f"Ready: {video_file.name} ({video_file.size/1e6:.1f} MB)")

        if st.button("🚀 Process with Both Models", type="primary", use_container_width=True):
            with st.spinner("Loading models (first time downloads ~40 MB each)..."):
                models = load_models()

            baseline_model = models.get("baseline")
            sam_model = models.get("sam")

            if baseline_model is None or sam_model is None:
                st.error("""
                Models not available. Upload them to Hugging Face first:
                1. Go to https://huggingface.co/sshivamvyas
                2. Create model repos: `ppe-compliance-baseline` and `ppe-compliance-sam`
                3. Upload `baseline_best.pt` and `best_sam_refined.pt` as `best.pt` in each
                """)
                st.stop()

            # ── Process Baseline ─────────────────────────────────────────
            st.markdown("### <span class='badge badge-base'>Baseline</span> Processing...", unsafe_allow_html=True)
            records_base, video_base, class_base, total_base = process_video(baseline_model, tmp_video, is_sam=False)

            # ── Process SAM ──────────────────────────────────────────────
            st.markdown("### <span class='badge badge-sam'>SAM-Refined</span> Processing...", unsafe_allow_html=True)
            records_sam, video_sam, class_sam, total_sam = process_video(sam_model, tmp_video, is_sam=True)

            # ── Results ──────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Results")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Baseline Detections", f"{total_base:,}", f"{total_base/max(len(records_base),1):.1f}/frame")
            with col2:
                st.metric("SAM-Refined Detections", f"{total_sam:,}", f"{total_sam/max(len(records_sam),1):.1f}/frame")

            # Charts
            all_cls = sorted(set(list(class_base.keys()) + list(class_sam.keys())))
            df = pd.DataFrame([{"Class":c,"Baseline":class_base.get(c,0),"SAM":class_sam.get(c,0)} for c in all_cls])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["Class"],y=df["Baseline"],name="Baseline",marker_color="#1a73e8"))
            fig.add_trace(go.Bar(x=df["Class"],y=df["SAM"],name="SAM-Refined",marker_color="#137333"))
            fig.update_layout(height=400, barmode="group", margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Annotated videos
            st.markdown("### 🎬 Annotated Videos")
            cv1, cv2 = st.columns(2)
            with cv1:
                st.markdown('<span class="badge badge-base">Baseline</span>', unsafe_allow_html=True)
                st.video(video_base)
            with cv2:
                st.markdown('<span class="badge badge-sam">SAM-Refined</span>', unsafe_allow_html=True)
                st.video(video_sam)

            # Download
            csv_data = pd.DataFrame(records_sam).to_csv(index=False)
            st.download_button("📥 Download Detection CSV", csv_data, "detections.csv", "text/csv")
