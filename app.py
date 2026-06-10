"""
PPE Compliance — Live Dashboard
================================
Upload video → GPU processes with Baseline & SAM-Teacher → compare side-by-side.
"""

import streamlit as st
import base64, requests
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
    .card { background:#1e1e2e; border-radius:12px; padding:20px; border:1px solid #333; color:#e0e0e0; }
    .card strong { color:#ffffff; font-size:0.95rem; }
    .comply-yes { color:#34a853; font-weight:700; }
    .comply-no { color:#ea4335; font-weight:700; }
</style>
""", unsafe_allow_html=True)

MODAL_URL = "https://sshivamvyas--ppe-compliance-detect-http.modal.run"
PPE_ITEMS = ["helmet","gloves","vest","boots","goggles"]

# ═══════════════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <h1>🦺 PPE Compliance Tracking System</h1>
    <p>
        Comparing <span class="badge badge-base">Baseline YOLO11m</span> (trained directly)
        vs <span class="badge badge-sam">SAM-Teacher YOLO11m</span> (9,994 SAM-refined labels) — <b>55% mAP improvement</b>
    </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### 📈 Model Performance (Test Set)")

st.sidebar.markdown("""
| Metric | Baseline | SAM | Δ |
|--------|:---:|:---:|:---:|
| mAP50 | 0.558 | **0.864** | +55% |
| mAP50-95 | 0.281 | **0.710** | +153% |
| Precision | 0.674 | **0.870** | +29% |
| Recall | 0.612 | **0.804** | +31% |
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Both YOLO11m** · 40.5 MB each  
Baseline: 86 epochs · 1.42 hrs  
SAM: 100 epochs · 1.44 hrs  
+66 min SAM labeling
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**⚡ Deployment**  
Streamlit Cloud (free)  
Modal T4 GPU (free credits)  
~10 sec per video
""")

# ═══════════════════════════════════════════════════════════════════════════════
#  Upload & Detect
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📤 Upload & Detect on GPU")

video_file = st.file_uploader("Upload video for live processing", type=["mp4","avi","mov","webm"],
                               help="Both models run simultaneously on T4 GPU — no model selection needed")

if video_file:
    if "last_video" not in st.session_state or st.session_state.last_video != video_file.name:
        st.session_state.last_video = video_file.name
        st.session_state.gpu_result = None

    st.video(video_file)
    st.caption(f"{video_file.name} — {video_file.size/1e6:.1f} MB")

    if st.button("🚀 Process on T4 GPU", type="primary", use_container_width=True):
        with st.spinner("⚡ Running both models on T4 GPU (~10-15 sec)..."):
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
                    data = resp.json()
                    if "error" in data:
                        st.error(f"Modal: {data['error'][:500]}")
                    else:
                        st.session_state.gpu_result = data
                        st.success(f"✅ {data['results']['baseline']['frames']} frames processed on T4 GPU")
                        st.balloons()
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # ── Show Results ────────────────────────────────────────────────────
    if "gpu_result" in st.session_state and st.session_state.gpu_result:
        r = st.session_state.gpu_result["results"]
        rb, rs = r["baseline"], r["sam"]

        # Check if SAM failed
        if "error" in rs:
            st.warning(f"SAM model encountered an error: {rs['error'][:200]}")
            st.stop()

        # ── KPI Row ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames", f"{rb['frames']}", f"{video_file.size/1e6:.1f}MB")
        c2.metric("Baseline Detections", f"{rb['total_detections']:,}",
                  f"{rb['total_detections']/max(rb['frames'],1):.1f}/frame")
        c3.metric("SAM-Teacher Detections", f"{rs['total_detections']:,}",
                  f"{rs['total_detections']/max(rs['frames'],1):.1f}/frame")
        c4.metric("Δ SAM vs Baseline", f"{rs['total_detections']-rb['total_detections']:+,}")

        st.caption(f"Processed on Modal T4 GPU · Both models same architecture & speed (~85 FPS on GPU)")

        # ── Compliance Summary Cards ─────────────────────────────────────
        st.markdown("### 🎯 Detection Breakdown: What's Present & Missing")

        def build_compliance(model_name, ct, total_frames):
            """Build a compliance card showing what PPE is detected vs missing."""
            rows = []
            for item in PPE_ITEMS:
                count = ct.get(item, 0)
                ratio = count / max(total_frames, 1)
                status = "✅" if ratio > 0.1 else "⚠️ Not detected"
                rows.append((item.capitalize(), f"{count:,}", f"{ratio:.1f}/frame", status))
            return rows

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown('<span class="badge badge-base">Baseline YOLO11m</span>', unsafe_allow_html=True)
            rows_b = build_compliance("baseline", rb["class_totals"], rb["frames"])
            st.dataframe(pd.DataFrame(rows_b, columns=["PPE Item","Detections","Density","Status"]),
                        hide_index=True, use_container_width=True)
        with cc2:
            st.markdown('<span class="badge badge-sam">SAM-Teacher YOLO11m</span>', unsafe_allow_html=True)
            rows_s = build_compliance("sam", rs["class_totals"], rs["frames"])
            st.dataframe(pd.DataFrame(rows_s, columns=["PPE Item","Detections","Density","Status"]),
                        hide_index=True, use_container_width=True)

        # ── Side-by-Side Bar Chart ──────────────────────────────────────
        st.markdown("### 📊 Per-Class Comparison")
        all_c = sorted(set(list(rb['class_totals'])+list(rs['class_totals'])))
        df = pd.DataFrame([{"Class":c,"Baseline":rb['class_totals'].get(c,0),"SAM":rs['class_totals'].get(c,0)} for c in all_c])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Class"],y=df["Baseline"],name="Baseline",marker_color="#1a73e8",text=df["Baseline"],textposition="outside"))
        fig.add_trace(go.Bar(x=df["Class"],y=df["SAM"],name="SAM-Teacher",marker_color="#137333",text=df["SAM"],textposition="outside"))
        fig.update_layout(height=400,barmode="group",margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig,use_container_width=True)

        # ── Annotated Preview ────────────────────────────────────────────
        st.markdown("### 📸 Annotated Preview (first frame)")
        cv1, cv2 = st.columns(2)
        with cv1:
            st.markdown('<span class="badge badge-base">Baseline</span>', unsafe_allow_html=True)
            if rb.get("preview_b64"):
                st.image(base64.b64decode(rb["preview_b64"]), use_container_width=True)
        with cv2:
            st.markdown('<span class="badge badge-sam">SAM-Teacher</span>', unsafe_allow_html=True)
            if rs.get("preview_b64"):
                st.image(base64.b64decode(rs["preview_b64"]), use_container_width=True)

        # ── Event Timeline ──────────────────────────────────────────────
        st.markdown("### 📈 Detection Timeline")
        base_ts = [(r["time"], r["detections"]) for r in rb.get("records", []) if r["detections"] > 0]
        sam_ts = [(r["time"], r["detections"]) for r in rs.get("records", []) if r["detections"] > 0]
        fig2 = go.Figure()
        if base_ts:
            t, d = zip(*base_ts)
            fig2.add_trace(go.Scatter(x=t,y=d,mode="lines",name="Baseline",line=dict(color="#1a73e8",width=1),opacity=0.7))
        if sam_ts:
            t, d = zip(*sam_ts)
            fig2.add_trace(go.Scatter(x=t,y=d,mode="lines",name="SAM-Teacher",line=dict(color="#137333",width=2),opacity=0.7))
        fig2.update_layout(height=300,xaxis_title="Seconds",yaxis_title="Detections/frame",margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2,use_container_width=True)

        # ── Download ────────────────────────────────────────────────────
        st.markdown("### 📥 Download Annotated Videos (15 sec)")
        dc1, dc2 = st.columns(2)
        with dc1:
            if rb.get("video_b64"):
                st.download_button("⬇ Baseline Annotated Video", base64.b64decode(rb["video_b64"]),
                                   "baseline_annotated.mp4", "video/mp4", use_container_width=True)
        with dc2:
            if rs.get("video_b64"):
                st.download_button("⬇ SAM-Teacher Annotated Video", base64.b64decode(rs["video_b64"]),
                                   "SAM_annotated.mp4", "video/mp4", use_container_width=True)

else:
    # No video uploaded yet — show placeholder
    st.info("👆 Upload a video above to see live PPE compliance detection. Both Baseline & SAM-Teacher models run automatically on a T4 GPU.")
