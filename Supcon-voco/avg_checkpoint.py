import torch
import os
from glob import glob
import sys

def load_checkpoint(filepath):
    return torch.load(filepath)

def average_checkpoints(checkpoint_files):
    avg_state_dict = None
    num_checkpoints = len(checkpoint_files)

    for checkpoint_file in checkpoint_files:
        state_dict = load_checkpoint(checkpoint_file)
        if avg_state_dict is None:
            avg_state_dict = {k: v.clone() for k, v in state_dict.items()}
        else:
            for k in avg_state_dict:
                avg_state_dict[k] += state_dict[k]

    for k in avg_state_dict:
        if avg_state_dict[k].dtype == torch.long:
            avg_state_dict[k] = avg_state_dict[k].float()
        avg_state_dict[k] /= num_checkpoints

    return avg_state_dict

def save_checkpoint(state_dict, filepath):
    torch.save(state_dict, filepath)

if __name__ == '__main__':
    # get the directory of the checkpoints from command line
    checkpoint_dir = sys.argv[1]
    # check if the function is called with correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python avg_checkpoint.py <checkpoint_dir>")
        sys.exit(1)
    # check the directory exists
    if not os.path.exists(checkpoint_dir):
        print("Directory not found")
        sys.exit(1)
    # get all the pt or pth files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') or f.endswith('.pth')]
    # sort as the modification time
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    # keep only the last 5 checkpoints
    checkpoint_files = checkpoint_files[-5:]
    # load and average the checkpoints
    avg_checkpoint = None
    avg_checkpoint = average_checkpoints([os.path.join(checkpoint_dir, f) for f in checkpoint_files])
    # save the averaged checkpoint
    save_checkpoint(avg_checkpoint, os.path.join(checkpoint_dir, 'avg.pth'))
    
