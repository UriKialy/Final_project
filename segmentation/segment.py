import os
import sys
import yaml
import json
import torch
from ultralytics import YOLO
from .masks_to_polygons import masks_to_polygons, split_train_test_val

# Path constants
TRAIN_ARGS_REL = os.path.join('runs', 'segment', 'train', 'args.yaml')
CFG_FILENAME = 'yolov8n-seg.yaml'  # or specify .pt weights

# -----------------------------------------------------------------------------
# 1) Data preparation: convert masks, split data, and write data.yaml for splits
# -----------------------------------------------------------------------------
def prepare_input():
    masks_to_polygons()
    split_train_test_val()
    write_data_yaml()


def write_data_yaml():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    # splitfolders output layout: data/train/images, data/val/images, data/test/images
    train_images = os.path.join(data_dir, 'train', 'images')
    val_images   = os.path.join(data_dir, 'val',   'images')
    test_images  = os.path.join(data_dir, 'test',  'images')

    os.makedirs(data_dir, exist_ok=True)
    data_cfg = {
        'train': train_images,
        'val':   val_images,
        'test':  test_images,
        'nc':    3,
        'names': ['benign', 'malignant', 'normal']
    }
    yaml_path = os.path.join(data_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_cfg, f)
    print(f'Wrote data.yaml → {yaml_path}')

# -----------------------------------------------------------------------------
# 2) Train: reuse previous args.yaml, override for GPU, doubling epochs & early stopping
#    then write a summary JSON at the end.
# -----------------------------------------------------------------------------
def train_model():
    assert torch.cuda.is_available(), 'CUDA unavailable!'

    # Dataset YAML
    script_dir = os.path.dirname(__file__)
    data_yaml  = os.path.join(script_dir, 'data', 'data.yaml')
    print(f'Starting training with dataset config: {data_yaml}')

    # Load prior training args
    repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    args_path = os.path.join(repo_root, TRAIN_ARGS_REL)
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f'Could not find args.yaml at {args_path}')
    with open(args_path) as f:
        args = yaml.safe_load(f)
    print('Original training args loaded')

    # Override: data, device, epochs (double), early stopping (patience)
    args['data'] = data_yaml
    args['device'] = 0
    orig_epochs = args.get('epochs', 100)
    args['epochs'] = orig_epochs * 2
    # if no patience provided, keep orig_epochs (so early stop = 100)
    args['patience'] = args.get('patience', orig_epochs)
    print(f"Epochs set to {args['epochs']} with early stopping patience {args['patience']}")

    # Instantiate model; let Ultralytics resolve YAML or .pt
    model_id = args.get('model', CFG_FILENAME)
    print(f'Loading model: {model_id}')
    model = YOLO(model_id)
    model.to('cuda:0')

    # Remove keys not accepted by .train()
    for k in ('model', 'mode'): args.pop(k, None)

    # Train
    results = model.train(**args)

    # After training, gather summary
    run_dir = getattr(model.trainer, 'save_dir', None)
    summary = {
        'run_dir': run_dir,
        'hyperparameters': {
            'epochs': args['epochs'],
            'patience': args['patience'],
            'batch': args.get('batch'),
            'imgsz': args.get('imgsz'),
            'optimizer': args.get('optimizer'),
            'lr0': args.get('lr0'),
            'lrf': args.get('lrf')
        },
        'weights': {
            'best': os.path.join(run_dir, 'weights', 'best.pt'),
            'last': os.path.join(run_dir, 'weights', 'last.pt')
        }
    }
    # Optionally extract final metrics if available
    if hasattr(results, 'metrics'):
        summary['metrics'] = results.metrics
    # Write summary JSON
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        summary_path = os.path.join(run_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Wrote summary → {summary_path}')
    else:
        print('Warning: could not determine run_dir to write summary')

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    prepare_input()
    train_model()

if __name__ == '__main__':
    main()
