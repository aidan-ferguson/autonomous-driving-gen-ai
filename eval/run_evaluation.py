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
import datetime
import shutil
import cv2
from ultralytics import YOLO
import torch
from numpy.typing import NDArray

from diffusion_model import DiffusionModel

# Cluster permissions
os.umask(0o002)

# For some reason FSOCO images are surrounded by a black border of thickness 140, we remove this
FSOCO_BORDER = 140

# How small either the width or height of bounding boxes can be and still be included in the training data (in pixels)
BB_THRESHOLD = 10

# What size of image should the YOLO network accept
YOLO_INPUT_SIZE = 256

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
  4: unknown
"""
    

def synthetic_data_schedule(trail_idx: int, real_world_train_size: int) -> int:
    """
    Get the number of synthetic samples we should add to the training set based on 
    the size of real world data in the train set

    :param trail_idx: What is the ID of the experiment being conducted
    :param real_world_train_size: The size of the set of *real world* training samples
    :returns: An integer indicated the amount of synthetic data to be added 
    """
    # For every subsequent trail add 50% of the initial set's worth of data
    return (real_world_train_size//2)*trail_idx

def train_yolo(epochs: int, data_dir: int):
    pass

def evaluate_diffusion_model(model_path: str, real_world_dir: str, sim_frame_dir: str, n_rw_samples: int) -> None:
    diffusion_model = DiffusionModel(model_path)

    # Make folder to house our evaluation results
    timestamp = datetime.datetime.now().strftime('%d.%m.%Y-%H.%M.%S')
    eval_dir = os.path.join(os.path.dirname(__file__), f"diffusion_evaluation_{timestamp}")
    dataset_dir = os.path.join(eval_dir, "dataset")
    train_image_dir = os.path.join(dataset_dir, "images")
    train_label_dir = os.path.join(dataset_dir, "labels")

    sim_mask_dir = os.path.join(sim_frame_dir, "masks")
    sim_label_dir = os.path.join(sim_frame_dir, "labels")

    if os.path.exists(eval_dir):
        raise Exception(f"Evaluation folder '{eval_dir}' already exists, quitting")
    else:
        os.mkdir(eval_dir)
        os.mkdir(os.path.join(eval_dir, "dataset"))
        os.mkdir(train_image_dir)
        os.mkdir(train_label_dir)
    
    # Get real world samples
    rw_image_dir = os.path.join(real_world_dir, "images")
    rw_label_dir = os.path.join(real_world_dir, "labels")
    rw_ids = [elem for elem in os.listdir(rw_image_dir) if os.path.isfile(os.path.join(rw_image_dir, elem))] 
    
    # Take N random images and find corresponding labels
    random.shuffle(rw_ids)
    rw_images = rw_ids[:n_rw_samples]
    rw_labels = [f"{'.'.join(image.split('.')[:-1])}.txt" for image in rw_images]

    # Copy the selected samples to the training folder in the generated evaluation folder
    for image, label in zip(rw_images, rw_labels):
        # FSOCO dataset has a border of 140px around the image - remove this 
        img = cv2.imread(os.path.join(rw_image_dir, image))
        # TODO: this might be incorrect as normalised values -> pixel space may require size of image with borders?
        img = img[FSOCO_BORDER:-FSOCO_BORDER, FSOCO_BORDER:-FSOCO_BORDER]
        cv2.imwrite(os.path.join(train_image_dir, image), img)

        # Note, we cannot just copy the annotation as the annotation is only valid for the full image (with borders)
        #   we must calculate a new label file
        with open(os.path.join(rw_label_dir, label), "r") as file:
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

            if (width*YOLO_INPUT_SIZE < BB_THRESHOLD) or (height*YOLO_INPUT_SIZE < BB_THRESHOLD):
                excluded_indices.append(ann_idx)

            old_label[ann_idx][1] = x1
            old_label[ann_idx][2] = y1
            old_label[ann_idx][3] = width
            old_label[ann_idx][4] = height

        # Now write out new file with corrected annotations
        with open(os.path.join(train_label_dir, label), "w") as file:
            label = [' '.join(list(map(str, ann))) for idx, ann in enumerate(old_label) if idx not in excluded_indices]
            file.write('\n'.join(label))

    synthetic_count = 0
    for idx in range(10):
        print(f"Evaluation step {idx}")

        new_synthetic_count = synthetic_data_schedule(idx, n_rw_samples)
        print(f"{new_synthetic_count=}")
        if (new_synthetic_count - synthetic_count) > 0:
            # We need to generate some images
            for sample_idx in range(synthetic_count, new_synthetic_count):
                # Generate an image label pair using simulator mask & sim bounding box info
                mask_path = os.path.join(sim_mask_dir, f"frame_{sample_idx}_mask.png")
                mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
                sample = diffusion_model.forward(mask, n_samples=1)[0]
                cv2.imwrite(os.path.join(train_image_dir, f"sampled_frame_{sample_idx}.png"), cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))

                # Copy over sampled frame bounding boxes but remove bounding boxes with width or height less than some threshold
                with open(os.path.join(sim_label_dir, f"frame_{sample_idx}_yolo.txt"), "r") as source_file:
                    with open(os.path.join(train_label_dir, f"sampled_frame_{sample_idx}.txt"), "r") as dest_file:
                        label = [list(map(float, line.split(' '))) for line in source_file.readlines()]
                        excluded_indices = []
                        for idx in range(list(label)):
                            label[idx][0] = int(label[idx][0])
                            if (label[idx][3]*YOLO_INPUT_SIZE < BB_THRESHOLD) or (label[idx][4]*YOLO_INPUT_SIZE < BB_THRESHOLD):
                                excluded_indices.append(idx)

                        label = [row for idx, row in enumerate(label) if idx not in excluded_indices]
                        dest_file.write('\n'.join([list(map(str, row)) for row in label]))


            synthetic_count = new_synthetic_count
        
        # YOLO training outputs a folder called 'runs' in the current working dir, so chdir into the iteration
        eval_step_dir = os.path.join(eval_dir, f"evaluation_step_{idx}")
        os.mkdir(eval_step_dir)
        os.chdir(eval_step_dir)

        # We can now train a YOLO network using the information we have so far
        model = YOLO('yolov8s.pt')
        yaml = generate_yolo_yaml(dataset_dir)

        with open(os.path.join(dataset_dir, "fsoco.yaml"), "w") as file:
            file.write(yaml)

        results = model.train(data=os.path.join(dataset_dir, "fsoco.yaml"), epochs=10, imgsz=YOLO_INPUT_SIZE, verbose=False)

        # We want to delete the generated labels.cache so the next iteration will read all the dataset
        os.remove(os.path.join(dataset_dir, "labels.cache"))


def main() -> None:
    parser = argparse.ArgumentParser(prog='Dissertation Evaluation Script')
    parser.add_argument('evaluation_type', choices=["gan", "diffusion"], help="Which type of network to evaluate")           
    parser.add_argument('model_path', help="Path to the model to be evaluated")
    parser.add_argument('real_world_dir', help="Directory containing the real world annotations in YOLO format")
    parser.add_argument('sim_frame_dir', help="Directory containing simulator frames used to inference the model being evaluated")
    parser.add_argument('--real_world_samples', default=100, help="How many real world samples should we include in the training dataset (default 100)")
    parser.add_argument('--random_seed', type=int, default=None, help="Random seed for the evaluation, defaults to no seed")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path) or not os.path.isfile(args.model_path):
        raise Exception(f"Path '{args.model_path}' does not exist or is not a file") 
    
    if not os.path.exists(args.real_world_dir):
        raise Exception(f"The real world directory '{args.real_world_dir}' does not exist!")
    
    if not os.path.exists(args.sim_frame_dir):
        raise Exception(f"The simulator frames dir '{args.sim_frame_dir}' does not exist!")

    if args.random_seed is not None:
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    if args.evaluation_type == "diffusion":
        print(f"Evaluating diffusion model {args.model_path}")
        evaluate_diffusion_model(args.model_path, args.real_world_dir, args.sim_frame_dir, args.real_world_samples)
    elif args.evaluation_type == "gan":
        print(f"Evaluating GAN model {args.model_path}")


if __name__ == "__main__":
    main()