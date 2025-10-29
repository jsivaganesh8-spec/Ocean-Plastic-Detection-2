import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

CLASSES = ["Plastics Trash"]


def voc_bbox_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int):
	cx = (xmin + xmax) / 2.0
	cy = (ymin + ymax) / 2.0
	w = xmax - xmin
	h = ymax - ymin
	return cx / img_w, cy / img_h, w / img_w, h / img_h


def convert_xml_file(xml_path: str) -> Tuple[int, int]:
	tree = ET.parse(xml_path)
	root = tree.getroot()
	img_w = int(root.find("size/width").text)
	img_h = int(root.find("size/height").text)

	labels: List[str] = []
	for obj in root.findall("object"):
		name = obj.find("name").text.strip()
		if name not in CLASSES:
			continue
		class_id = CLASSES.index(name)
		bbox = obj.find("bndbox")
		xmin = float(bbox.find("xmin").text)
		ymin = float(bbox.find("ymin").text)
		xmax = float(bbox.find("xmax").text)
		ymax = float(bbox.find("ymax").text)
		x, y, w, h = voc_bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
		labels.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

	txt_path = os.path.splitext(xml_path)[0] + ".txt"
	with open(txt_path, "w", encoding="utf-8") as f:
		f.write("\n".join(labels))

	return len(labels), 1


def run_split(split_dir: str) -> Tuple[int, int]:
	xml_files = glob.glob(os.path.join(split_dir, "*.xml"))
	total_boxes = 0
	total_files = 0
	for xml_path in xml_files:
		boxes, files = convert_xml_file(xml_path)
		total_boxes += boxes
		total_files += files
	return total_boxes, total_files


def main():
	root = Path(__file__).parent
	for split in ["train", "valid", "test"]:
		split_dir = str(root / split)
		if not os.path.isdir(split_dir):
			print(f"Skip missing split: {split_dir}")
			continue
		tb, tf = run_split(split_dir)
		print(f"Converted {tf} files with {tb} boxes in: {split}")

	with open(root / "classes.txt", "w", encoding="utf-8") as f:
		for name in CLASSES:
			f.write(name + "\n")
	print("Classes written to classes.txt")

	yaml_path = root / "ocean_debris.yaml"
	yaml = f"""
path: {root.as_posix()}
train: train
val: valid
names:
  0: Plastics Trash
""".strip()
	with open(yaml_path, "w", encoding="utf-8") as f:
		f.write(yaml)
	print(f"Dataset YAML written to {yaml_path}")


if __name__ == "__main__":
	main()
