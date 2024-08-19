import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

CSV_HEADER = ["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "val/box_loss", "val/cls_loss", "val/dfl_loss", "lr/pg0", "lr/pg1", "lr/pg2"]

def main():
    parser = argparse.ArgumentParser(prog='Dissertation Evaluation Script')
    parser.add_argument('evaluation_folders', type=list[str], nargs='+', help="List of evaluation folders from which to generate summary graphs")
    args = parser.parse_args()

    # Mapping from evaluation name to list of CSV's
    evaluation_data = {}

    # For every folder, go through every evaluation step and load the training CSV
    for eval_folder in [''.join(folder) for folder in args.evaluation_folders]:
        if not os.path.exists(eval_folder):
            raise Exception(f"Path '{eval_folder}' does not exist")
        base_dir = os.path.basename(eval_folder.rstrip("/"))

        evaluation_data[base_dir] = []
        
        eval_step_dirs = [os.path.join(eval_folder, eval_step_dir) for eval_step_dir in os.listdir(eval_folder) if os.path.isdir(os.path.join(eval_folder, eval_step_dir)) and eval_step_dir.startswith("evaluation_step")] 
        eval_step_dirs = sorted(eval_step_dirs, key=lambda dir: int(dir.split("_")[-1]))
        for eval_step in eval_step_dirs:
            with open(os.path.join(eval_step, "runs/detect/train/results.csv"), "r") as file:
                # Skip header, convert everything to a numpy array and store
                data = file.readlines()[1:]
                data = np.array([[float(elem) for elem in row.split(",")] for row in data])

            evaluation_data[base_dir].append(data)
        
        evaluation_data[base_dir] = np.array(evaluation_data[base_dir])

    print(evaluation_data)


if __name__ == "__main__":
    main()