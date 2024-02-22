# script for evaluating the effectiveness of supplementing YOLO training datasets with synthetic data
# points of note:
#   the real world folder must be structure like:
#       root
#         > images                          (e.g. AMZ_0001.png)
#         > labels  (in .txt YOLO format)   (e.g. AMZ_0001.txt)
#
#   the simulator data folder must be structure like:
#       root
#         > images                          (e.g. frame_0.png)
#         > masks                           (e.g. frame_0_mask.png)
#         > labels (in .txt YOLO format)    (e.g. frame_0_yolo.txt)
#

import argparse
import os
import random
import cv2
from ultralytics import YOLO
from numpy.typing import NDArray

# Cluster permissions
os.umask(0o002)

# For some reason FSOCO images are surrounded by a black border of thickness 140, we remove this
FSOCO_BORDER = 140

# How small either the width or height of bounding boxes can be and still be included in the training data (in pixels)
BB_THRESHOLD = 5

# What size of image should the YOLO network accept
YOLO_INPUT_SIZE = 640

def generate_yolo_yaml(dataset_dir: str):
    return f"""
path: {dataset_dir}
train: images
val: images
test:

names:
  0: blue
  1: orange
  2: largeorange
  3: yellow
"""

def train_yolo(src_dataset_dir: str, epochs: int) -> None:
    # Make folder to house the dataset results
    dataset_dir = os.path.join(os.path.dirname(__file__), "yolo_dataset")
    os.mkdir(dataset_dir)
    
    train_image_dir = os.path.join(dataset_dir, "images")
    train_label_dir = os.path.join(dataset_dir, "labels")

    os.mkdir(train_image_dir)
    os.mkdir(train_label_dir)

    # Get real world samples
    src_image_dir = os.path.join(src_dataset_dir, "images")
    src_label_dir = os.path.join(src_dataset_dir, "labels")
    
    images = [elem for elem in os.listdir(src_image_dir) if os.path.isfile(os.path.join(src_image_dir, elem))] 
    labels = [f"{'.'.join(image.split('.')[:-1])}.txt" for image in images]

    print(images)
    print(labels)

    # Copy the selected samples to the new YOLO dataset, removing FSOCO border as we go
    for image, label in zip(images, labels):
        # FSOCO dataset has a border of 140px around the image - remove this 
        img = cv2.imread(os.path.join(src_image_dir, image))
        img = img[FSOCO_BORDER:-FSOCO_BORDER, FSOCO_BORDER:-FSOCO_BORDER]
        cv2.imwrite(os.path.join(train_image_dir, image), img)

        # Note, we cannot just copy the annotation as the annotation is only valid for the full image (with borders)
        #   we must calculate a new label file
        with open(os.path.join(src_label_dir, label), "r") as file:
            old_label = [list(map(float, line.split(' '))) for line in file.readlines()]

        excluded_indices = []

        # Multiply out by full image size, then re-normalise
        for ann_idx in range(len(old_label)):
            # Change class back to int
            old_label[ann_idx][0] = int(old_label[ann_idx][0])

            # Convert from old image normalised space to cropped image pixel space
            x1 = (old_label[ann_idx][1] * (img.shape[1] + (FSOCO_BORDER*2))) - FSOCO_BORDER
            y1 = (old_label[ann_idx][2] * (img.shape[0] + (FSOCO_BORDER*2))) - FSOCO_BORDER
            width = (old_label[ann_idx][3] * (img.shape[1] + (FSOCO_BORDER*2)))
            height = (old_label[ann_idx][4] * (img.shape[0] + (FSOCO_BORDER*2)))

            # Re-normalise in cropped image space
            x1 /= img.shape[1]
            y1 /= img.shape[0]
            width /= img.shape[1]
            height /= img.shape[0]

            # if (width*YOLO_INPUT_SIZE < BB_THRESHOLD) or (height*YOLO_INPUT_SIZE < BB_THRESHOLD):
            #     excluded_indices.append(ann_idx)

            old_label[ann_idx][1] = x1
            old_label[ann_idx][2] = y1
            old_label[ann_idx][3] = width
            old_label[ann_idx][4] = height

        # Now write out new file with corrected annotations
        with open(os.path.join(train_label_dir, label), "w") as file:
            label = [' '.join(list(map(str, ann))) for idx, ann in enumerate(old_label) if idx not in excluded_indices]
            file.write('\n'.join(label))

        # We can now train a YOLO network using the information we have so far
        model = YOLO('yolov8x.pt')
        yaml = generate_yolo_yaml(dataset_dir)

        with open(os.path.join(dataset_dir, "fsoco.yaml"), "w") as file:
            file.write(yaml)

        model.train(data=os.path.join(dataset_dir, "fsoco.yaml"), epochs=epochs, imgsz=YOLO_INPUT_SIZE, verbose=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog='FSOCO YOLO training script')
    parser.add_argument('fsoco_dir', help="Directory containing the FSOCO dataset in YOLO format")
    parser.add_argument('--epochs', default=150, type=int, help="Number of epochs to train for")
    args = parser.parse_args()
    
    if not os.path.exists(args.fsoco_dir):
        raise Exception(f"The real world directory '{args.fsoco_dir}' does not exist!")

    train_yolo(args.fsoco_dir, args.epochs)
    
if __name__ == "__main__":
    main()