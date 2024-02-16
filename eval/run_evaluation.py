# script for evaluating the effectiveness of supplementing YOLO training datasets with synthetic data
import argparse
import os

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

def evaluate_diffusion_model(real_world_dir: str) -> None:
    # Setup synthetic folder
    synthetic_folder = os.path.join(os.path.dirname(__file__), "synthetic_diffusion_input")
    
    if not os.path.exists(synthetic_folder):
        os.mkdir(synthetic_folder)
    
    for idx in range(10):
        synthetic_count = synthetic_data_schedule(idx, 10)
        


def main() -> None:
    parser = argparse.ArgumentParser(prog='Dissertation Evaluation Script')
    parser.add_argument('evaluation_type', choices=["gan", "diffusion"], help="Which type of network to evaluate")           
    parser.add_argument('model_path', help="Path to the model to be evaluated")
    parser.add_argument('real_world_dir', help="Directory containing the real world annotations in YOLO format")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path) or not os.path.isfile(args.model_path):
        raise Exception(f"Path '{args.model_path}' does not exist or is not a file") 
    
    if not os.path.exits(args.real_world_dir):
        raise Exception(f"The real world directory '{args.real_world_dir}' does not exist!")

    if args.evaluation_type == "diffusion":
        print(f"Evaluating diffusion model {args.model_path}")
        evaluate_diffusion_model(args.real_world_dir)
    elif args.evaluation_type == "gan":
        print(f"Evaluating GAN model {args.model_path}")


if __name__ == "__main__":
    main()