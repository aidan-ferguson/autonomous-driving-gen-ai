# Script for parsing the FSOCO segmentation dataset format into the format that we use to train the network
import argparse
import os
import cv2
from tqdm import tqdm
import zlib
import base64
import numpy as np
import json

# For some reason FSOCO images are surrounded by a black border of thickness 140, we remove this
FSOCO_BORDER = 140

# Note, this is in BGR format - it tells us how classes are mapped to colours
class_colour_map = {
    "seg_yellow_cone": np.array([255,255,0]),
    "seg_blue_cone": np.array([0,0,255]),
    "seg_orange_cone": np.array([255,165,0]),
    "seg_large_orange_cone": np.array([255,69,0]),
    "seg_unknown_cone": np.array([100,100,100])
}

def base64_2_mask(s):
    """
    bitmap to binary mask function from https://developer.supervisely.com/api-references/supervisely-annotation-json-format/objects
    """
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

def binary_mask_to_colour(bitmap, class_title):
    """
    Take a binary mask and produce the corresponding BGR image for it
    """
    indices = np.nonzero(bitmap)
    coloured = np.zeros((bitmap.shape[0], bitmap.shape[1], 3))
    coloured[indices] = class_colour_map[class_title][::-1]
    return coloured

def parse_team_folder(fsoco_folder, output_folder, team_folder) -> None:
    """
    Iterate over one team's folder, processing and saving the images and masks to the output folder
    """
    img_folder = os.path.join(fsoco_folder, team_folder, "img")
    ann_folder = os.path.join(fsoco_folder, team_folder, "ann")
    if not os.path.exists(img_folder) or not os.path.exists(ann_folder):
        print(f"img or ann not found in team folder: {team_folder}")
        return None
    
    # Get all images
    images = os.listdir(img_folder)

    # Copy image files to output folder, prepending team name and removing border at bottom of image
    for filename in images:
        # Copy actual image
        img = cv2.imread(os.path.join(img_folder, filename))
        img = img[FSOCO_BORDER:-FSOCO_BORDER, FSOCO_BORDER:-FSOCO_BORDER]
        filename_extension_less = '.'.join(filename.split(".")[:-1])
        cv2.imwrite(os.path.join(output_folder, f"{team_folder}_{filename_extension_less}.jpg"), img)

        # Generate and save corresponding mask image file
        with open(os.path.join(ann_folder, f"{filename}.json"), "r") as file:
            ann = json.loads(file.read())
            mask = np.zeros_like(img)

            # For every object, get the mask image and place it into the overall image mask
            for object in ann["objects"]:
                bitmap = base64_2_mask(object["bitmap"]["data"])
                bitmap_nonzero = np.nonzero(bitmap)
                origin = list(map(lambda x: x - FSOCO_BORDER, object["bitmap"]["origin"]))
                
                # This is long but all it is doing is placing the coloured mask into the overall mask image - the 'nonzero' index ensures we can have overlapping cones
                #  by ensuring that all zero pixels in the boolean bitmap are skipped over
                mask[origin[1]:origin[1]+bitmap.shape[0], origin[0]:origin[0]+bitmap.shape[1]][bitmap_nonzero] = binary_mask_to_colour(bitmap, object["classTitle"])[bitmap_nonzero]
            cv2.imwrite(os.path.join(output_folder, f"{team_folder}_{filename_extension_less}_mask.png"), mask)

def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(prog='FSOCO Dataset Parser')
    parser.add_argument('fsoco_folder', help="The input FSOCO folder containing the segmentation annotated data")
    parser.add_argument('output_folder', help="The parsed dataset will be placed in this folder")
    args = parser.parse_args()
    
    if not os.path.exists(args.fsoco_folder) or not os.path.isdir(args.fsoco_folder):
        raise Exception("The folder you provided does not exist")
    
    if not os.path.exists(os.path.join(args.fsoco_folder, "meta.json")):
        raise Exception("The folder you provided is not a valid FSOCO dataset - meta.json is missing")
    
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    
    # Get all the folders in the FSOCO dataset
    folders = [elem for elem in os.listdir(args.fsoco_folder) if os.path.isdir(os.path.join(args.fsoco_folder, elem))]
    
    for folder in tqdm(folders):
        parse_team_folder(args.fsoco_folder, args.output_folder, folder)

if __name__ == "__main__":
    main()
