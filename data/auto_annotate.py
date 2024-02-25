# Script for auto-annotating real world data and producing semantic segmentation masks
import argparse
import os
from ultralytics.data.annotator import auto_annotate
import cv2
import numpy as np
from tqdm import tqdm

from fsoco_parser import class_colour_map

CLASS_NAME_MAP = {
    0: "seg_blue_cone",
    1: "seg_orange_cone",
    2: "seg_large_orange_cone",
    3: "seg_yellow_cone",
}

def label_to_mask(image_dir: str, auto_annotate_dir: str, label: str) -> None:
    """
    Convert an auto-annotation label to an image mask
    """

    # Read annotation and image
    with open(os.path.join(auto_annotate_dir, label)) as file:
        annotations = [list(map(float, line.strip().split(" "))) for line in file.read().strip().split("\n")]

    img_filename = f"{'.'.join(label.split('.')[:-1])}.png"
    img = cv2.imread(os.path.join(image_dir, img_filename))

    # Generate semantic map image
    mask = np.zeros_like(img)
    for annotation in annotations:        
        points = annotation[1:]
        points = np.reshape(np.array(points), (len(points)//2, 2))

        # Little bit of a hack, but don't process the points if in the bottom center of image (car decal sometimes gets picked up as a cone)
        center = np.mean(points, axis=0)
        if center[0] > 0.45 and center[0] < 0.65 and center[1] > 0.8:
            continue
        
        points[:, 0] *= img.shape[1]
        points[:, 1] *= img.shape[0]
        points = np.int32(points)

        colour = tuple(reversed(list(map(int, class_colour_map[CLASS_NAME_MAP[int(annotation[0])]]))))
        mask = cv2.fillPoly(mask, [points], colour)


    mask_filename = f"{'.'.join(label.split('.')[:-1])}_mask.png"
    cv2.imwrite(os.path.join(image_dir, mask_filename), mask)
        

def main() -> None:
    parser = argparse.ArgumentParser(prog='automatic cone segmentation')
    parser.add_argument('images_dir', help="Path to the folder with images to be annotated")
    parser.add_argument('yolo_model', help="Path to the YOLOv8 models for cone detections")   
    args = parser.parse_args()

    if not os.path.exists(args.images_dir) or not os.path.isdir(args.images_dir):
        raise Exception("Image folder does not exist")


    if not os.path.exists(args.yolo_model):
        raise Exception("YOLO model path does not exist")
    
    for file in os.listdir(args.images_dir):
        if 'mask' in file:
            print("Warning: there may be previous masks files in the image directory, please delete all the image masks before proceeding!")

    # auto_annotate(args.images_dir, det_model=args.yolo_model, sam_model='sam_b.pt')
    auto_annotate_dir = os.path.join(os.path.dirname(args.images_dir), f"{os.path.basename(args.images_dir.rstrip('/'))}_auto_annotate_labels")

    for label in tqdm(os.listdir(auto_annotate_dir)):
        label_to_mask(args.images_dir, auto_annotate_dir, label)


if __name__ == "__main__":
    main()
