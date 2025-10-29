import argparse
from pathlib import Path
from ultralytics import YOLO


DEFAULT_MODEL = "yolov8n.pt"


def main():
	parser = argparse.ArgumentParser(description="Train YOLOv8 on ocean debris dataset")
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="base model or weights path")
	parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
	parser.add_argument("--imgsz", type=int, default=640, help="image size")
	parser.add_argument("--batch", type=int, default=16, help="batch size")
	parser.add_argument("--device", type=str, default="0", help="device: 0 for first CUDA GPU")
	args = parser.parse_args()

	root = Path(__file__).parent
	yaml_path = root / "ocean_debris.yaml"
	if not yaml_path.exists():
		raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}. Run prepare_dataset.py first.")

	model = YOLO(args.model)
	results = model.train(
		data=str(yaml_path),
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		project=str(root / "runs"),
		verbose=True,
	)
	print(results)

	best = root / "runs" / "detect" / "train" / "weights" / "best.pt"
	if best.exists():
		model = YOLO(str(best))
		metrics = model.val(data=str(yaml_path), imgsz=args.imgsz, device=args.device)
		print(metrics)
	else:
		print("Best weights not found after training.")


if __name__ == "__main__":
	main()
