#!/usr/bin/env python3
import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# --------------------------------------------------------
# Utility: GPU probe with pynvml (optional)
# --------------------------------------------------------
def probe_gpus():
    gpus = []
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='ignore')
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024**2)
            gpus.append(f"GPU {i}: {name} ({mem} MiB)")
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return gpus

# --------------------------------------------------------
# Load labels + deterministic colors
# --------------------------------------------------------
def load_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def get_class_color(idx):
    rng = np.random.RandomState(idx * 999)  # deterministic per class
    return tuple(int(x) for x in rng.randint(0, 255, 3))

# --------------------------------------------------------
# Device + provider selection
# --------------------------------------------------------
def select_providers(request):
    available = ort.get_available_providers() if hasattr(ort, "get_available_providers") else []
    chosen = []
    notes = []

    if request == "auto":
        if "TensorrtExecutionProvider" in available:
            chosen = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            notes.append("🎮 Using GPU (TensorRT + CUDA)")
        elif "CUDAExecutionProvider" in available:
            chosen = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            notes.append("🎮 Using GPU (CUDA)")
        else:
            chosen = ["CPUExecutionProvider"]
            notes.append("🖥️  Using CPU only")
    elif request == "cuda":
        chosen = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        notes.append("🎮 Using GPU (CUDA)")
    elif request == "tensorrt":
        chosen = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        notes.append("🎮 Using GPU (TensorRT + CUDA)")
    else:
        chosen = ["CPUExecutionProvider"]
        notes.append("🖥️  Using CPU only")

    return chosen, notes

# --------------------------------------------------------
# Letterbox resize
# --------------------------------------------------------
def letterbox(img, new_shape=(640, 640), color=(114,114,114)):
    shape = img.shape[:2]  # h, w
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw /= 2; dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img = cv2.copyMakeBorder(img, top,bottom,left,right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

# --------------------------------------------------------
# Inference + postprocessing
# --------------------------------------------------------
def run_inference(args):
    labels = load_labels(args.labels)
    os.makedirs(args.output, exist_ok=True)

    # Header
    console_header = []
    console_header.append("="*60)
    console_header.append("YOLO ONNX Object Detection")
    console_header.append("="*60)
    console_header.append(f"📦 Model: {args.model}")
    console_header.append(f"🏷️  Labels: {args.labels}")
    console_header.append(f"📁 Input: {args.input}")
    console_header.append(f"💾 Output: {args.output}")
    console_header.append(f"🎯 Confidence Threshold: {args.threshold}")
    console_header.append(f"🔄 NMS Threshold: {args.nms}")
    console_header.append(f"📐 Padding: {args.pad}px")
    console_header.append(f"🖥️  Device Request: {args.cuda}")
    console_header.append("="*60)
    print("\n".join(console_header))

    # GPU probe
    available = ort.get_available_providers() if hasattr(ort, "get_available_providers") else []
    gpu_info = probe_gpus()
    print("\n🔍 Checking GPU availability...")
    print(f"Available providers: {available}")
    for g in gpu_info:
        print(f"✅ {g}")

    # Select providers
    providers, notes = select_providers(args.cuda)
    for n in notes: print(n)

    # Try to create session (handle provider failures gracefully)
    session = None
    tried = []
    for prov_list in [providers, ["CUDAExecutionProvider","CPUExecutionProvider"], ["CPUExecutionProvider"]]:
        try:
            tried.append(prov_list)
            session = ort.InferenceSession(args.model, providers=prov_list)
            providers = session.get_providers()
            break
        except Exception as e:
            # Don't spam the console with long ONNX tracebacks; print concise message
            print(f"⚠️  Could not initialize session with {prov_list}: {str(e).splitlines()[0]}")
            session = None
    if session is None:
        print("❌ Failed to create an ONNX Runtime session with any provider. Exiting.")
        return

    # Inputs
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"✓ Loaded model: {args.model}")
    print(f"✓ Model input shape: {input_shape}")
    # In some ONNXs the spatial dims may be None; handle defensively:
    try:
        h_in, w_in = input_shape[2], input_shape[3]
        print(f"✓ Model input size: {h_in}x{w_in}")
    except Exception:
        print(f"✓ Model input size: (unknown)")

    print(f"✓ Runtime providers: {session.get_providers()}")

    # Collect images (recursive search)
    def collect_image_files(input_path):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        found = []
        if os.path.isdir(input_path):
            for root, _, filenames in os.walk(input_path):
                for fn in sorted(filenames):
                    if fn.lower().endswith(exts):
                        found.append(os.path.join(root, fn))
        else:
            # single file passed as input
            if input_path.lower().endswith(exts):
                found = [input_path]
        return found

    files = collect_image_files(args.input)
    print(f"\n📁 Found {len(files)} images to process (searched recursively)")

    # Prepare output dirs
    annotated_dir = os.path.join(args.output, "annotated")
    crops_dir = os.path.join(args.output, "crops")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    # Progress file path (so external worker can poll it)
    progress_path = os.path.join(args.output, "progress.json")
    try:
        # initialize progress file
        with open(progress_path, "w") as pf:
            json.dump({"processed": 0, "total": len(files)}, pf)
    except Exception:
        pass

    # Stats
    total_dets = 0
    class_counts = {lbl:0 for lbl in labels}
    t0 = time.time()

    # Loop images
    for i, path in enumerate(sorted(files)):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # update processed count anyway
            try:
                with open(progress_path, "w") as pf:
                    json.dump({"processed": i+1, "total": len(files)}, pf)
            except Exception:
                pass
            continue

        # Keep a pristine original for crops (no drawing)
        orig = img_bgr.copy()
        # Separate image for annotations
        ann = img_bgr.copy()

        img, r, dwdh = letterbox(orig)
        img_in = img.transpose(2,0,1)[None].astype(np.float32)/255.0

        # Run model
        try:
            outputs = session.run([out_name], {in_name: img_in})[0]
        except Exception as e:
            print(f"⚠️  Inference failed on {os.path.basename(path)}: {e}")
            # update progress on failure of this image
            try:
                with open(progress_path, "w") as pf:
                    json.dump({"processed": i+1, "total": len(files)}, pf)
            except Exception:
                pass
            continue

        # Process detections
        base_name = os.path.splitext(os.path.basename(path))[0]
        for det in outputs:
            if len(det) < 7:
                continue
            _, x0,y0,x1,y1,cls_id,score = det[:7]
            if score < args.threshold:
                continue
            cls_id = int(cls_id)
            if cls_id < 0 or cls_id >= len(labels):
                cls = f"class_{cls_id}"
            else:
                cls = labels[cls_id]

            # scale back
            box = np.array([x0,y0,x1,y1], dtype=float)
            box -= np.array(dwdh*2)
            box /= r
            box = box.round().astype(int)
            x0i,y0i,x1i,y1i = box
            x0c,y0c = max(0,x0i), max(0,y0i)
            x1c,y1c = min(orig.shape[1]-1,x1i), min(orig.shape[0]-1,y1i)
            if x1c <= x0c or y1c <= y0c:
                continue

            # Color per class
            color = get_class_color(cls_id)

            # Draw only on 'ann' (annotated image) — not on orig
            cv2.rectangle(ann, (x0c,y0c), (x1c,y1c), color, 2)
            label_text = f"{cls} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(ann, (x0c, y0c - th - 4), (x0c + tw, y0c), color, -1)
            cv2.putText(ann, label_text, (x0c, y0c - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Crop from pristine original (clean crop)
            crop = orig[y0c:y1c, x0c:x1c]
            if crop.size == 0:
                continue

            # Ensure unique crop filename using running class count +1
            idx = class_counts.get(cls, 0) + 1
            class_dir = os.path.join(crops_dir, cls)
            os.makedirs(class_dir, exist_ok=True)
            crop_name = f"{base_name}_{cls}_{idx}.jpg"
            crop_path = os.path.join(class_dir, crop_name)
            cv2.imwrite(crop_path, crop)

            # update counts AFTER saving (so idx reflects that saved file)
            class_counts[cls] = idx
            total_dets += 1

        # Save annotated image (with boxes)
        ann_out = os.path.join(annotated_dir, os.path.basename(path))
        cv2.imwrite(ann_out, ann)

        # update progress file
        try:
            with open(progress_path, "w") as pf:
                json.dump({"processed": i+1, "total": len(files)}, pf)
        except Exception:
            pass

    t1 = time.time()
    elapsed = t1 - t0
    fps = (len(files) / elapsed) if elapsed > 0 else 0.0

    # Build console summary
    console_summary = []
    console_summary.append("\n" + "="*60)
    console_summary.append("✅ Detection Complete!")
    console_summary.append("="*60)
    console_summary.append(f"⏱️  Processing time: {elapsed:.2f} seconds")
    console_summary.append(f"🚀 Average FPS: {fps:.2f}")
    console_summary.append(f"📊 Total detections: {total_dets}")
    console_summary.append(f"📁 Output directory: {args.output}")
    console_summary.append("\n📈 Detection counts by class:")
    # sort by descending counts for nicer reading
    for cls, cnt in sorted(class_counts.items(), key=lambda x: (-x[1], x[0])):
        console_summary.append(f"   • {cls}: {cnt}")
    console_summary.append(f"\n📄 Report saved to: {os.path.join(args.output,'detection_report.json')}")
    console_summary.append(f"📝 Summary saved to: {os.path.join(args.output,'detection_summary.txt')}")
    console_summary.append(f"🖼️  Annotated images: {annotated_dir}")
    console_summary.append(f"✂️  Cropped objects: {crops_dir}")
    console_summary.append("="*60)
    print("\n".join(console_summary))

    # Save JSON report
    report_json = os.path.join(args.output, "detection_report.json")
    with open(report_json, "w") as f:
        json.dump({
            "model": args.model,
            "labels": labels,
            "providers": session.get_providers(),
            "files": len(files),
            "elapsed": elapsed,
            "fps": fps,
            "total_detections": total_dets,
            "class_counts": class_counts
        }, f, indent=2)

    # Write the full console header + summary into detection_summary.txt
    report_txt = os.path.join(args.output, "detection_summary.txt")
    with open(report_txt, "w") as f:
        f.write("\n".join(console_header) + "\n\n")
        f.write("\n".join(console_summary) + "\n")

# --------------------------------------------------------
# CLI
# --------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="output_"+time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--pad", type=int, default=10)
    ap.add_argument("--cuda", default="auto", choices=["auto","cpu","cuda","tensorrt"])
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)

