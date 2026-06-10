"""Upload models to Modal Volume."""
import modal

# Upload to Modal's persistent storage
vol = modal.Volume.from_name("ppe-models", create_if_missing=True)

with vol.batch_upload() as batch:
    batch.put_file("models/baseline_best.pt", "/baseline_best.pt")
    batch.put_file("models/best_sam_refined.pt", "/best_sam_refined.pt")

print("Models uploaded to Modal Volume!")
