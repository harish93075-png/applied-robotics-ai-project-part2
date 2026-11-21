"""Build the submission folder: the Webots package zip (Assignments 4 and 5) next to the report."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "submission"
ZIP = OUT_DIR / "8757HIQT_HarishAshokkumar_Webots.zip"
TOP = "8757HIQT_HarishAshokkumar_Webots"

FILES = [
    "README.md",
    "webots/worlds/colour_block_search.wbt",
    "webots/worlds/cnn_object_test.wbt",
    "webots/controllers/epuck_controller/epuck_controller.py",
    "webots/controllers/epuck_controller/resnet9_int8.tflite",
]
EVIDENCE = {  # path in the project -> path inside the zip
    "docs/block_search_final.jpg": "evidence/block_search_final.jpg",
    "docs/block_search_frames.png": "evidence/block_search_camera_frames.png",
    "docs/webots_live_cnn_final.jpg": "evidence/cnn_tour_live.jpg",
    "webots/controllers/epuck_controller/captures/tour_log.txt": "evidence/cnn_tour_log.txt",
}
EVIDENCE.update({str(p.relative_to(ROOT)).replace("\\", "/"): f"evidence/block_search_{p.name}"
                 for p in sorted((ROOT / "docs/block_search_runs").glob("run_*.txt"))})

OUT_DIR.mkdir(exist_ok=True)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(ROOT / f, f"{TOP}/{f}")
    for src, dst in EVIDENCE.items():
        z.write(ROOT / src, f"{TOP}/{dst}")
with zipfile.ZipFile(ZIP) as z:
    for info in z.infolist():
        print(f"{info.file_size:>10,}  {info.filename}")
print("wrote", ZIP, f"({ZIP.stat().st_size / 1e6:.1f} MB)")
