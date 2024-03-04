# Script for parsing the FSOCO segmentation dataset format into the format that we use to train the network
import argparse
import os
import shutil
import json

LABEL_ID_MAP = {
    "blue": 0,
    "orange": 1,
    "large_orange": 2,
    "yellow": 3
}

def convert_vott_folder(vott_folder, image_names, json_filename, output_folder) -> None:
    """
    Convert a vott folder into YOLO format
    """
    img_folder = os.path.join(output_folder, "images")
    label_folder = os.path.join(output_folder, "labels")
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    if not os.path.exists(label_folder):
        os.mkdir(label_folder)
    
    # Copy image files to output folder
    for image_name in image_names:
        shutil.copyfile(os.path.join(vott_folder, image_name), os.path.join(img_folder, image_name))

    # Convert from VoTT format to YOLO annotation format
    with open(os.path.join(vott_folder, json_filename), "r") as file:
        data = json.loads(file.read())
        for asset_id in data["assets"]:
            asset = data["assets"][asset_id]
            image_filename = asset["asset"]["name"]
            image_width = asset["asset"]["size"]["width"]
            image_height = asset["asset"]["size"]["height"]

            yolo_annotations = []
            for ann in asset["regions"]:
                 class_id = LABEL_ID_MAP[ann["tags"][0]]
                 top_left = (ann["boundingBox"]["left"], ann["boundingBox"]["top"])
                 ann_width = ann["boundingBox"]["width"]
                 ann_height = ann["boundingBox"]["height"]

                 center_x = top_left[0] + (ann_width / 2)
                 center_y = top_left[1] + (ann_height / 2)

                 center_x /= image_width
                 center_y /= image_height
                 ann_width /= image_width
                 ann_height /= image_height

                 yolo_annotations.append(f"{class_id} {center_x} {center_y} {ann_width} {ann_height}")

            yolo_filename = '.'.join(image_filename.split(".")[:-1]) + ".txt"
            with open(os.path.join(label_folder, yolo_filename), "w") as file:
                file.write('\n'.join(yolo_annotations))

def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='VoTT to YOLO converter')
    parser.add_argument('vott_folder', help="The exported VoTT folder")
    parser.add_argument('output_folder', help="The parsed dataset will be placed in this folder")
    args = parser.parse_args()
    
    if not os.path.exists(args.vott_folder) or not os.path.isdir(args.vott_folder):
        raise Exception("The folder you provided does not exist")
    
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    
    # Get all the folders in the FSOCO dataset
    images = [elem for elem in os.listdir(args.vott_folder) if elem.endswith(".png")]
    jsons = [elem for elem in os.listdir(args.vott_folder) if elem.endswith(".json")]

    assert len(jsons) == 1
    json_file = jsons[0]

    convert_vott_folder(args.vott_folder, images, json_file, args.output_folder)


if __name__ == "__main__":
    main()
