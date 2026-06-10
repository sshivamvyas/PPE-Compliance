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

CLASS_NAMES = {0:"helmet", 1:"gloves", 2:"vest", 3:"boots", 4:"goggles", 6:"Person"}
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
    """
    GPU inference with person-PPE association via pose keypoints.
    Returns per-person compliance for both models.
    """
    import cv2, numpy as np, json, tempfile, base64, traceback, subprocess, shutil
    from collections import defaultdict
    from datetime import datetime
    from pathlib import Path
    import torch
    from ultralytics import YOLO

    # ── Pose keypoint indices (COCO format) ───────────────────────────
    NOSE, L_EYE, R_EYE = 0, 1, 2
    L_SHOULDER, R_SHOULDER = 5, 6
    L_WRIST, R_WRIST = 9, 10
    L_ANKLE, R_ANKLE = 15, 16

    # PPE-to-keypoints mapping for spatial association
    PPE_KEYPOINTS = {
        0: [NOSE, L_EYE, R_EYE],      # helmet → head
        1: [L_WRIST, R_WRIST],         # gloves → hands
        2: [L_SHOULDER, R_SHOULDER],   # vest → torso
        3: [L_ANKLE, R_ANKLE],         # boots → feet
        4: [L_EYE, R_EYE],             # goggles → eyes
    }
    PPE_NAMES = {0:"helmet",1:"gloves",2:"vest",3:"boots",4:"goggles"}
    KEYPOINT_NAMES = {0:"nose",1:"l_eye",2:"r_eye",5:"l_shoulder",6:"r_shoulder",9:"l_wrist",10:"r_wrist",15:"l_ankle",16:"r_ankle"}
    PROXIMITY_PX = 120  # max px distance from keypoint to PPE center

    try:
        # ── Load models ──────────────────────────────────────────────
        b = Path(MODEL_PATH) / "baseline_best.pt"
        s = Path(MODEL_PATH) / "best_sam_refined.pt"
        if not b.exists() or not s.exists():
            vol_files = list(Path(MODEL_PATH).glob("**/*"))
            return {"error": f"Models missing. Volume: {[str(f) for f in vol_files]}"}

        print("[Modal] Loading baseline + SAM models...")
        model_base = YOLO(str(b))
        model_sam = YOLO(str(s))

        print("[Modal] Loading YOLO-pose model (auto-downloads ~5MB)...")
        model_pose = YOLO("yolo11n-pose.pt")

        # ── Decode video ─────────────────────────────────────────────
        video_bytes = base64.b64decode(item["video_b64"])
        video_name = item.get("video_name", "video.mp4")
        tmp_video = tempfile.mktemp(suffix=f"_{video_name}")
        with open(tmp_video, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(tmp_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Modal] Video: {total} frames, {w}x{h}, {fps}fps")

        # Output settings
        max_dim = 720
        scale = min(max_dim / max(w, h), 1.0)
        out_w, out_h = int(w * scale), int(h * scale)
        preview_frames = min(total, int(fps * 15))
        POSE_EVERY_N = 3  # run pose on every 3rd frame for speed

        # ── Process both models ──────────────────────────────────────
        results = {}
        for model_name, det_model in [("baseline", model_base), ("sam", model_sam)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            is_sam = (model_name == "sam")

            # Output video: use ffmpeg for reliable browser-compatible encoding
            frame_dir = tempfile.mkdtemp()
            out_video = tempfile.mktemp(suffix=".mp4")

            records, class_totals, total_dets = [], {}, 0
            compliance_log = []  # per-person per-frame
            pose_cache = None

            print(f"[Modal] Processing {model_name}...")

            for idx in range(total):
                ret, frame = cap.read()
                if not ret: break

                # Detection
                preds = det_model(frame, verbose=False, device=0, imgsz=640)[0]

                boxes, cls_ids, confs = np.array([]), np.array([], dtype=int), np.array([])
                if preds.boxes is not None:
                    boxes = preds.boxes.xyxy.cpu().numpy()
                    cls_ids = preds.boxes.cls.cpu().numpy().astype(int)
                    confs = preds.boxes.conf.cpu().numpy()
                    if is_sam:
                        cls_ids = np.array([SAM_REMAP.get(int(c), int(c)) for c in cls_ids])

                # Pose (every Nth frame)
                person_boxes = []
                person_kpts = []
                if idx % POSE_EVERY_N == 0:
                    pose_preds = model_pose(frame, verbose=False, device=0, imgsz=640)[0]
                    if pose_preds.boxes is not None and pose_preds.keypoints is not None:
                        p_boxes = pose_preds.boxes.xyxy.cpu().numpy()
                        kpts = pose_preds.keypoints.xy.cpu().numpy()
                        for pi in range(len(p_boxes)):
                            if float(pose_preds.boxes.conf[pi]) > 0.4:
                                person_boxes.append(p_boxes[pi])
                                person_kpts.append(kpts[pi])
                    pose_cache = (person_boxes, person_kpts)
                elif pose_cache:
                    person_boxes, person_kpts = pose_cache

                # ── Person-PPE association ───────────────────────────
                per_person = []
                for pi, pbox in enumerate(person_boxes):
                    px1, py1, px2, py2 = map(int, pbox)
                    ppe_status = {}
                    for ppe_id in [0,1,2,3,4]:  # helmet,gloves,vest,boots,goggles
                        found = False
                        for di in range(len(boxes)):
                            if int(cls_ids[di]) != ppe_id: continue
                            bx1,by1,bx2,by2 = boxes[di]
                            bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
                            # Check if PPE center is near relevant keypoints
                            for kp_idx in PPE_KEYPOINTS.get(ppe_id, []):
                                if kp_idx < len(person_kpts[pi]):
                                    kx, ky = person_kpts[pi][kp_idx]
                                    if kx > 0 and ky > 0:
                                        dist = np.sqrt((bcx-kx)**2 + (bcy-ky)**2)
                                        if dist < PROXIMITY_PX:
                                            found = True
                                            break
                            if found: break
                        ppe_status[PPE_NAMES[ppe_id]] = found

                    # Also try IoU-based fallback for any unmatched PPE
                    for ppe_id in [0,1,2,3,4]:
                        if ppe_status.get(PPE_NAMES[ppe_id], False): continue
                        for di in range(len(boxes)):
                            if int(cls_ids[di]) != ppe_id: continue
                            bx1,by1,bx2,by2 = boxes[di]
                            iou_x1 = max(px1, bx1); iou_y1 = max(py1, by1)
                            iou_x2 = min(px2, bx2); iou_y2 = min(py2, by2)
                            if iou_x2 > iou_x1 and iou_y2 > iou_y1:
                                iou_area = (iou_x2-iou_x1)*(iou_y2-iou_y1)
                                ppe_area = (bx2-bx1)*(by2-by1)
                                if ppe_area > 0 and iou_area/ppe_area > 0.3:
                                    ppe_status[PPE_NAMES[ppe_id]] = True
                                    break

                    per_person.append({"id": pi+1, "compliance": ppe_status})

                # Frame record
                rec = {"frame":idx,"time":round(idx/fps,2),"detections":int(len(boxes)),"classes":{}}
                for cid in cls_ids:
                    cn = CLASS_NAMES.get(int(cid), f"c{int(cid)}")
                    rec["classes"][cn] = rec["classes"].get(cn, 0) + 1
                    class_totals[cn] = class_totals.get(cn, 0) + 1
                total_dets += len(boxes)
                records.append(rec)

                if per_person:
                    compliance_log.append({"frame":idx,"time":round(idx/fps,2),"persons":per_person})

                # ── Draw preview frame ───────────────────────────────
                if idx < preview_frames:
                    disp = frame.copy()

                    # Draw person boxes + compliance status
                    for pi, pbox in enumerate(person_boxes):
                        px1,py1,px2,py2 = map(int, pbox)
                        # Green if all 5 PPE present, yellow if partial, red if none
                        if pi < len(per_person):
                            compliant_count = sum(per_person[pi]["compliance"].values())
                            if compliant_count == 5: color = (0,255,0)
                            elif compliant_count >= 3: color = (0,255,255)
                            else: color = (0,0,255)
                        else:
                            color = (0,255,0)

                        cv2.rectangle(disp, (px1,py1), (px2,py2), color, 2)
                        status_text = f"P{pi+1}: {compliant_count}/5" if pi < len(per_person) else f"P{pi+1}"
                        cv2.putText(disp, status_text, (px1, py1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Draw PPE detections
                    for i in range(len(boxes)):
                        x1,y1,x2,y2 = map(int, boxes[i]); cid = int(cls_ids[i])
                        color = COLORS.get(cid, (128,128,128))
                        cv2.rectangle(disp, (x1,y1), (x2,y2), color, 1)
                        name = CLASS_NAMES.get(cid, f"c{cid}")
                        cv2.putText(disp, f"{name} {confs[i]:.2f}", (x1,y1-3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                    if scale < 1.0:
                        disp = cv2.resize(disp, (out_w, out_h))
                    if idx < preview_frames:
                        cv2.imwrite(f"{frame_dir}/frame_{idx:06d}.png", disp)

            cap.release()

            # Encode frames to MP4 using ffmpeg (reliable browser-compatible output)
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", f"{frame_dir}/frame_%06d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "ultrafast", "-crf", "28",
                out_video
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True)
            shutil.rmtree(frame_dir, ignore_errors=True)

            # Aggregate compliance
            person_compliance_agg = defaultdict(lambda: defaultdict(int))
            total_person_frames = 0
            for entry in compliance_log:
                total_person_frames += 1
                for p in entry["persons"]:
                    pid = p["id"]
                    for item_name, present in p["compliance"].items():
                        if present:
                            person_compliance_agg[pid][item_name] += 1

            compliance_summary = {}
            for pid in sorted(person_compliance_agg.keys()):
                d = person_compliance_agg[pid]
                compliance_summary[f"Person {pid}"] = {
                    item: round(d[item] / max(total_person_frames, 1) * 100, 1)
                    for item in ["helmet","gloves","vest","boots","goggles"]
                }

            with open(out_video, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode()

            results[model_name] = {
                "total_detections": total_dets, "frames": total,
                "class_totals": class_totals, "records": records,
                "video_b64": video_b64,
                "compliance": compliance_summary,
            }
            print(f"[Modal] {model_name}: {total_dets} dets, {len(compliance_summary)} persons tracked")

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
