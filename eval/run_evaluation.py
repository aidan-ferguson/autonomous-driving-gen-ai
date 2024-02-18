# script for evaluating the effectiveness of supplementing YOLO training datasets with synthetic data
# points of note:
#   the real world folder must be structure like:
#       root
#         > images
#         > labels (in .txt YOLO format)

import argparse
import os
import random

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

def evaluate_diffusion_model(real_world_dir: str, n_rw_samples: int) -> None:
    # Setup synthetic folder
    synthetic_folder = os.path.join(os.path.dirname(__file__), "synthetic_diffusion_input")
    
    if not os.path.exists(synthetic_folder):
        os.mkdir(synthetic_folder)
    
    # Get real world samples
    rw_image_dir = os.path.join(real_world_dir, "images")
    rw_label_dir = os.path.join(real_world_dir, "labels")
    rw_ids = [elem for elem in os.listdir(rw_image_dir) if os.path.isfile(os.path.join(rw_image_dir, elem))]
    # real_world_samples = 

    random.shuffle(rw_ids)

    print(rw_ids[:10])

    # for idx in range(10):
    #     synthetic_count = synthetic_data_schedule(idx, 10)
        


def main() -> None:
    parser = argparse.ArgumentParser(prog='Dissertation Evaluation Script')
    parser.add_argument('evaluation_type', choices=["gan", "diffusion"], help="Which type of network to evaluate")           
    parser.add_argument('model_path', help="Path to the model to be evaluated")
    parser.add_argument('real_world_dir', help="Directory containing the real world annotations in YOLO format")
    parser.add_argument('--real_world_samples', default=100, help="How many real world samples should we include in the training dataset")
    parser.add_argument('--random_seed', type=int, default=None, help="Random seed for the evaluation, defaults to no seed")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path) or not os.path.isfile(args.model_path):
        raise Exception(f"Path '{args.model_path}' does not exist or is not a file") 
    
    if not os.path.exits(args.real_world_dir):
        raise Exception(f"The real world directory '{args.real_world_dir}' does not exist!")

    if args.random_seed is not None:
        random.seed(args.random_seed)
        # TODO: torch random seed

    if args.evaluation_type == "diffusion":
        print(f"Evaluating diffusion model {args.model_path}")
        evaluate_diffusion_model(args.real_world_dir, args.real_world_samples)
    elif args.evaluation_type == "gan":
        print(f"Evaluating GAN model {args.model_path}")


if __name__ == "__main__":
    main()