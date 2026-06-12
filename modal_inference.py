"""
PPE Compliance — Modal GPU Inference
======================================
T4 GPU inference: upload a video → both models process it → get back annotated video + JSON.

Setup:
  1. modal setup            (one-time browser auth)
  2. modal run modal_inference.py::upload_models   (upload .pt files)
  3. modal deploy modal_inference.py               (deploy GPU function)

Dashboard usage:
  import modal
  detect = modal.Function.lookup("ppe-compliance", "detect")
  result = detect.remote(video_bytes, video_name)
"""

import modal
import os
from pathlib import Path

# ── Image (GPU environment) ──────────────────────────────────────────────────

image = (
    modal.Image.debian_slim()
    .run_commands("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    .pip_install(
        "ultralytics>=8.3.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "fastapi",
    )
    .apt_install("ffmpeg")
)

app = modal.App("ppe-compliance", image=image)

# ── Persistent Volume ────────────────────────────────────────────────────────

MODEL_VOLUME = modal.Volume.from_name("ppe-models", create_if_missing=True)
MODEL_PATH = "/models"

# ── Config ───────────────────────────────────────────────────────────────────

CLASS_NAMES = {0:"helmet", 1:"gloves", 2:"vest", 3:"boots", 4:"goggles", 5:"none",
               6:"Person", 7:"no_helmet", 8:"no_goggle", 9:"no_gloves", 10:"no_boots"}
SAM_REMAP = {0:6, 1:0, 2:2, 3:1, 4:4, 5:3}
COLORS = {0:(255,255,0), 1:(255,0,255), 2:(0,165,255), 3:(255,0,0), 4:(0,255,255), 6:(0,255,0)}

# ═══════════════════════════════════════════════════════════════════════════════
#  GPU Inference
# ═══════════════════════════════════════════════════════════════════════════════

@app.cls(gpu="T4", volumes={MODEL_PATH: MODEL_VOLUME}, timeout=600)
class PPEDetector:
    @modal.enter()
    def load(self):
        import torch
        from ultralytics import YOLO
        print("[Modal] Loading models on T4 GPU...")
        b = Path(MODEL_PATH) / "baseline_best.pt"
        s = Path(MODEL_PATH) / "best_sam_refined.pt"
        self.model_base = YOLO(str(b)) if b.exists() else None
        self.model_sam = YOLO(str(s)) if s.exists() else None
        print(f"[Modal] Baseline: {'OK' if self.model_base else 'MISSING'}, SAM: {'OK' if self.model_sam else 'MISSING'}")

    @modal.method()
    def detect(self, video_bytes: bytes, video_name: str = "video.mp4") -> dict:
        """Run both models on uploaded video. Returns dict with results."""
        import cv2
        import numpy as np
        import json
        import tempfile
        import base64
        from datetime import datetime

        if self.model_base is None or self.model_sam is None:
            return {"error": "Models not loaded. Run upload_models first."}

        # Save video to temp file
        tmp_video = tempfile.mktemp(suffix=f"_{video_name}")
        with open(tmp_video, "wb") as f:
            f.write(video_bytes)

        results = {}
        for model_name, model in [("baseline", self.model_base), ("sam", self.model_sam)]:
            cap = cv2.VideoCapture(tmp_video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_video = tempfile.mktemp(suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

            records, class_totals, total_dets = [], {}, 0
            is_sam = (model_name == "sam")

            for idx in range(total):
                ret, frame = cap.read()
                if not ret: break

                preds = model(frame, verbose=False, device=0)[0]

                if preds.boxes is None:
                    writer.write(frame)
                    records.append({"frame":idx, "time":round(idx/fps,2), "detections":0, "classes":{}})
                    continue

                boxes = preds.boxes.xyxy.cpu().numpy()
                cls = preds.boxes.cls.cpu().numpy().astype(int)
                confs = preds.boxes.conf.cpu().numpy()

                if is_sam:
                    cls = np.array([SAM_REMAP.get(int(c), int(c)) for c in cls])

                # Annotate frame
                disp = frame.copy()
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    cid = int(cls[i])
                    color = COLORS.get(cid, (128,128,128))
                    cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
                    name = CLASS_NAMES.get(cid, f"c{cid}")
                    cv2.putText(disp, f"{name} {confs[i]:.2f}", (x1,y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                writer.write(disp)

                rec = {"frame":idx, "time":round(idx/fps,2), "detections":int(len(boxes)), "classes":{}}
                for cid in cls:
                    cn = CLASS_NAMES.get(int(cid), f"c{int(cid)}")
                    rec["classes"][cn] = rec["classes"].get(cn, 0) + 1
                    class_totals[cn] = class_totals.get(cn, 0) + 1
                total_dets += len(boxes)
                records.append(rec)

            cap.release(); writer.release()

            with open(out_video, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode()

            results[model_name] = {
                "total_detections": total_dets,
                "frames": total,
                "class_totals": class_totals,
                "records": records,
                "video_b64": video_b64,
            }

        return {"results": results, "processed_at": datetime.now().isoformat()}

# ── Upload Models ────────────────────────────────────────────────────────────

@app.function(
    volumes={MODEL_PATH: MODEL_VOLUME},
)
def upload_models():
    """Upload both .pt files to Modal Volume. Run once after deploy — requires models/ dir to be included in mount."""
    import shutil

    # Models are expected to be mounted alongside this script
    base = Path(__file__).parent / "models"
    local_models = [
        str(base / "baseline_best.pt"),
        str(base / "best_sam_refined.pt"),
    ]

    for src_path in local_models:
        src = Path(src_path)
        if not src.exists():
            print(f"SKIP: {src} not found")
            continue
        dst = Path(MODEL_PATH) / src.name
        shutil.copy(str(src), str(dst))
        print(f"Uploaded {src.name} ({dst.stat().st_size / 1e6:.1f} MB)")

    print("Done. Models available:")
    for f in Path(MODEL_PATH).glob("*.pt"):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

# ── HTTP Endpoint (no Modal SDK needed on client) ────────────────────────────

@app.function(
    gpu="T4",
    volumes={MODEL_PATH: MODEL_VOLUME},
    timeout=600,
    allow_concurrent_inputs=1,
)
@modal.fastapi_endpoint(method="POST")
def detect_http(item: dict):
    """HTTP endpoint: runs both Baseline & SAM models on uploaded video. Returns annotated previews + JSON metrics."""
    import cv2, numpy as np, json, tempfile, base64, traceback, subprocess, shutil
    from datetime import datetime
    from pathlib import Path
    from ultralytics import YOLO

    try:
        # ── Load models ──────────────────────────────────────────────
        b_path = Path(MODEL_PATH) / "baseline_best.pt"
        s_path = Path(MODEL_PATH) / "best_sam_refined.pt"
        print(f"[Modal] baseline: {b_path.exists()}, sam: {s_path.exists()}")

        m_base = YOLO(str(b_path))
        m_sam = YOLO(str(s_path))

        # ── Decode video ─────────────────────────────────────────────
        video_bytes = base64.b64decode(item["video_b64"])
        vname = item.get("video_name", "video.mp4")
        tmp_video = tempfile.mktemp(suffix=f"_{vname}")
        with open(tmp_video, "wb") as f:
            f.write(video_bytes)

        # ── Process both models ──────────────────────────────────────
        results = {}

        for model_name, model in [("baseline", m_base), ("sam", m_sam)]:
            try:
                cap = cv2.VideoCapture(tmp_video)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[Modal] {model_name}: {total}frames {w}x{h} @{fps}fps")

                max_dim = 720
                scale = min(max_dim / max(w, h), 1.0)
                ow, oh = int(w * scale), int(h * scale)
                is_sam = (model_name == "sam")

                records, class_totals, total_dets = [], {}, 0
                preview_b64 = ""  # first annotated frame as JPEG
                max_download_frames = min(total, int(fps * 15))  # 15 sec for download
                out_video = tempfile.mktemp(suffix=".mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_video, fourcc, fps, (ow, oh))

                for idx in range(total):
                    ret, frame = cap.read()
                    if not ret: break

                    preds = model(frame, verbose=False, device=0, imgsz=640)[0]
                    disp = frame.copy()

                    if preds.boxes is not None:
                        boxes = preds.boxes.xyxy.cpu().numpy()
                        cls = preds.boxes.cls.cpu().numpy().astype(int)
                        confs = preds.boxes.conf.cpu().numpy()
                        if is_sam:
                            cls = np.array([SAM_REMAP.get(int(c), int(c)) for c in cls])

                        for i in range(len(boxes)):
                            x1,y1,x2,y2 = map(int, boxes[i])
                            cid = int(cls[i])
                            color = COLORS.get(cid, (128,128,128))
                            cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
                            name = CLASS_NAMES.get(cid, f"c{cid}")
                            cv2.putText(disp, f"{name} {confs[i]:.2f}", (x1,y1-4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                        rec = {"frame":idx, "time":round(idx/fps,2), "detections":int(len(boxes)), "classes":{}}
                        for cid in cls:
                            cn = CLASS_NAMES.get(int(cid), f"c{int(cid)}")
                            rec["classes"][cn] = rec["classes"].get(cn, 0) + 1
                            class_totals[cn] = class_totals.get(cn, 0) + 1
                        total_dets += len(boxes)
                        records.append(rec)
                    else:
                        records.append({"frame":idx,"time":round(idx/fps,2),"detections":0,"classes":{}})
                        # Write raw frame (no detections)
                        if idx < max_download_frames:
                            if scale < 1.0:
                                writer.write(cv2.resize(disp, (ow, oh)))
                            else:
                                writer.write(disp)

                    # Save first annotated frame as preview JPEG
                    if idx == 0:
                        if scale < 1.0:
                            disp = cv2.resize(disp, (ow, oh))
                        _, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        preview_b64 = base64.b64encode(buf).decode()

                    # Write frame to output video (first 15 sec only)
                    if idx < max_download_frames:
                        if scale < 1.0:
                            writer.write(cv2.resize(disp, (ow, oh)))
                        else:
                            writer.write(disp)

                cap.release(); writer.release()

                # Read output video
                video_b64 = ""
                if Path(out_video).exists():
                    with open(out_video, "rb") as f:
                        data = f.read()
                    if len(data) > 1000:
                        video_b64 = base64.b64encode(data).decode()

                results[model_name] = {
                    "total_detections": total_dets, "frames": total,
                    "class_totals": class_totals, "records": records,
                    "preview_b64": preview_b64,
                    "video_b64": video_b64,
                }
                print(f"[Modal] {model_name} DONE: {total_dets} dets, video={len(video_b64)/1024:.0f}KB")

            except Exception as e:
                print(f"[Modal] {model_name} FAILED: {traceback.format_exc()}")
                results[model_name] = {"error": str(e), "total_detections": 0, "frames": 0}

        return {"results": results, "processed_at": datetime.now().isoformat()}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"}


# ── Verify ──────────────────────────────────────────────────────────────────

@app.function(volumes={MODEL_PATH: MODEL_VOLUME})
def check():
    """Check models in volume."""
    p = Path(MODEL_PATH)
    if p.exists():
        for f in p.glob("*.pt"):
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
    else:
        print("No models found. Run upload_models first.")
