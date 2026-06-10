# 🦺 PPE Compliance Detection

**SAM-Teacher → YOLO-Student Pipeline for construction PPE compliance.**

---

## Models

| | Baseline | SAM-Refined |
|---|:-:|:-:|
| **mAP50** | 0.558 | **0.864** (+55%) |
| **mAP50-95** | 0.281 | **0.710** (+153%) |
| **Classes** | 11 | 6 |
| **Size** | 40.5 MB | 40.5 MB |

## Pipeline

```
Raw Images → Grounding DINO → SAM 1 (Masks) → Tight YOLO BBoxes → YOLO11m Student
```

- **Teacher:** Grounding DINO + SAM 1 auto-labels with pixel-perfect masks
- **Student:** YOLO11m trained on refined labels (same hyperparams)
- **Result:** 55% mAP50 improvement over direct YOLO training

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space
2. Choose **Streamlit** SDK
3. Upload these files or connect to this repo
4. Space auto-deploys with free T4 GPU

## Folder Structure

```
ppe-deploy-phase/
├── app.py              ← Unified dashboard (entry point)
├── models/
│   ├── baseline_best.pt      ← Baseline model (mAP50=0.558)
│   └── best_sam_refined.pt   ← SAM-Refined model (mAP50=0.864)
├── requirements.txt
├── packages.txt        ← System deps for HF Spaces
└── README.md
```
