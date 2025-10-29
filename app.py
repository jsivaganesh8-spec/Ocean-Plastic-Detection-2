import os
from pathlib import Path
from typing import List, Dict

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO

APP_TITLE = "Ocean Debris Detector"
CLASS_NAMES = ["Plastics Trash"]


@st.cache_resource
def load_model(weights_path: Path | None) -> YOLO:
	if weights_path and weights_path.exists():
		return YOLO(str(weights_path))
	# fallback to pretrained and hope class generalization; still useful for demo
	return YOLO("yolov8n.pt")


def draw_detections(image: np.ndarray, boxes: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> np.ndarray:
	out = image.copy()
	for i in range(len(boxes)):
		x1, y1, x2, y2 = boxes[i].astype(int)
		c = float(conf[i])
		label_idx = int(cls[i])
		label = CLASS_NAMES[label_idx] if 0 <= label_idx < len(CLASS_NAMES) else str(label_idx)
		cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
		cv2.putText(out, f"{label} {c:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
	return out


def summarize_detections(image_shape: tuple, boxes: np.ndarray, cls: np.ndarray) -> Dict:
	h, w = image_shape[:2]
	areas = []
	for i in range(len(boxes)):
		x1, y1, x2, y2 = boxes[i]
		area = max(0, (x2 - x1)) * max(0, (y2 - y1))
		areas.append(area)
	pixel_area = h * w
	coverage = float(sum(areas) / pixel_area) if pixel_area > 0 else 0.0
	counts = int(len(boxes))
	return {"count": counts, "coverage": coverage}


def run_image_inference(model: YOLO, image: np.ndarray, conf_thres: float):
	res = model.predict(source=image, stream=False, conf=conf_thres, verbose=False)[0]
	if res.boxes is None or len(res.boxes) == 0:
		return image, pd.DataFrame(), {"count": 0, "coverage": 0.0}
	boxes_xyxy = res.boxes.xyxy.cpu().numpy()
	conf = res.boxes.conf.cpu().numpy()
	cls = res.boxes.cls.cpu().numpy()
	vis = draw_detections(image, boxes_xyxy, conf, cls)

	records = []
	for i in range(len(boxes_xyxy)):
		x1, y1, x2, y2 = boxes_xyxy[i]
		records.append({
			"label": CLASS_NAMES[int(cls[i])] if int(cls[i]) < len(CLASS_NAMES) else str(int(cls[i])),
			"confidence": float(conf[i]),
			"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)
		})
	df = pd.DataFrame.from_records(records)
	summary = summarize_detections(image.shape, boxes_xyxy, cls)
	return vis, df, summary


def main():
	st.set_page_config(page_title=APP_TITLE, page_icon="🌊", layout="wide")
	st.title(APP_TITLE)
	st.caption("Detect and analyze plastic debris using your trained model.")

	root = Path(__file__).parent
	best_weights = root / "runs" / "detect" / "train" / "weights" / "best.pt"
	model = load_model(best_weights if best_weights.exists() else None)

	with st.sidebar:
		st.header("Settings")
		conf_thres = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
		mode = st.radio("Mode", ["Image", "Video", "Webcam"], index=0)
		st.markdown("Model weights:")
		st.code(str(best_weights if best_weights.exists() else "yolov8n.pt"), language="text")

	if mode == "Image":
		upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
		if upload is not None:
			arr = np.frombuffer(upload.read(), np.uint8)
			image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			vis, df, summary = run_image_inference(model, image, conf_thres)
			col1, col2 = st.columns([2, 1])
			with col1:
				st.image(vis, caption="Detections", use_column_width=True)
			with col2:
				st.metric("Plastic items detected", summary["count"])
				st.metric("Estimated coverage", f"{summary['coverage']*100:.2f}%")
				if not df.empty:
					st.dataframe(df, use_container_width=True, height=300)

	elif mode == "Video":
		upload = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
		if upload is not None:
			# Write to temp file for OpenCV
			tmp_path = Path(st.experimental_get_query_params().get("tmp", ["tmp"])[0])
			tmp_path.mkdir(exist_ok=True)
			vid_file = tmp_path / "input_video.mp4"
			with open(vid_file, "wb") as f:
				f.write(upload.read())
			cap = cv2.VideoCapture(str(vid_file))
			frame_placeholder = st.empty()
			stats_placeholder = st.empty()
			counts = []
			coverages = []
			while cap.isOpened():
				ret, frame = cap.read()
				if not ret:
					break
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				vis, _, summary = run_image_inference(model, frame, conf_thres)
				counts.append(summary["count"])
				coverages.append(summary["coverage"])
				frame_placeholder.image(vis, use_column_width=True)
				stats_placeholder.caption(f"Avg count: {np.mean(counts):.2f} | Avg coverage: {np.mean(coverages)*100:.2f}%")
			cap.release()

	else:  # Webcam
		st.info("Press Stop to end the webcam stream.")
		cap = cv2.VideoCapture(0)
		frame_placeholder = st.empty()
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			vis, _, _ = run_image_inference(model, frame, conf_thres)
			frame_placeholder.image(vis, use_column_width=True)
		cap.release()

	st.divider()
	st.subheader("Dataset quicklook")
	cols = st.columns(3)
	for idx, split in enumerate(["train", "valid", "test"]):
		split_dir = root / split
		if split_dir.exists():
			jpgs = list(split_dir.glob("*.jpg"))[:1]
			if jpgs:
				img = cv2.cvtColor(cv2.imread(str(jpgs[0])), cv2.COLOR_BGR2RGB)
				cols[idx].image(img, caption=f"{split}: {jpgs[0].name}")


if __name__ == "__main__":
	main()
