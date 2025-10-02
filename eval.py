import json
import os
import sys
import argparse

import torch

from src.data.iwood.iwood_utils import generate_batch_coco_annotation

# ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch or single-image inference script, with optional plotting and result output"
    )
    parser.add_argument('--config', '-c', type=str,
                        default='configs/iwood/iwood_r50vd_6x_coco.yml',
                        help='Path to the config file')
    parser.add_argument('--resume', '-r', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable AMP')
    parser.add_argument('--tuning', '-t', type=str,
                        help='Tuning mode, if any')
    parser.add_argument('--seed', type=int, default=66,
                        help='Random seed')

    # Only use --image_dir argument, supports both file and directory
    parser.add_argument('--image_dir', type=str, required=True,
                        help='If a single image file, process one image; if a directory, recursively process all images inside')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (when processing a single image)')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Whether to draw and save images with detection boxes')
    parser.add_argument('--print_results', action='store_true', default=False,
                        help='Whether to print file names and predicted class list')
    parser.add_argument('--print_output', action='store_true', default=False,
                        help='Print full inference output')
    return parser.parse_args()


def gather_images(path: str):
    """
    If path is a file, return a list containing this file;
    If path is a directory, recursively traverse and return all image file paths.
    """
    if os.path.isfile(path):
        return [path]
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = []
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            if os.path.splitext(fn.lower())[1] in exts:
                images.append(os.path.join(root, fn))
    return images



def main():
    args = parse_args()

    # Initialize distributed training and random seed
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    # Collect single or multiple images
    image_list = gather_images(args.image_dir)  # data/IWOOD-archaeology-20250413/梓木/C5-7_20250408074040883.jpg

    # If only one image and the input is a file, set root_dir to its parent directory
    if len(image_list) == 1 and os.path.isfile(image_list[0]):
        root_dir = os.path.dirname(image_list[0])
    else:
        # For multiple images, use their common root directory
        root_dir = os.path.commonpath(image_list)

    temp_json = generate_batch_coco_annotation(image_list, root_dir)

    # Load config and modify dynamically
    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )
    cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'] = root_dir
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = temp_json
    cfg.yaml_cfg['val_dataloader']['batch_size'] = 1
    cfg.yaml_cfg['output_dir'] = args.output_dir

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and run inference on all images at once
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    results = solver.val_single_pic(
        print_results = args.print_results,
        plot = args.plot,
        output_dir = args.output_dir,
    )
    # print("===results:",results)

    # Print results and draw images
    if args.print_results:
        for res in results:
            fname = res['file_name']
            labels = res['image_label']

            # Convert to Python list
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy().tolist()
            print(f"[{fname}] --> labels: {labels}")

    if args.print_output:
            print("results:",results)

    # Delete temporary file
    try:
        os.remove(temp_json)
    except OSError:
        pass


if __name__ == '__main__':
    main()
