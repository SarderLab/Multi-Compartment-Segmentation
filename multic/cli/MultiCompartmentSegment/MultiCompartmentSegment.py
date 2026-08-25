import argparse
import os
import sys

sys.path.append('..')
from segmentationschool.segmentation_school import run_it
import torch

from storage_client import StorageClient

# retire-girder-dependency: no more Slicer CLI XML / ctk_cli argument parsing — the dispatcher (see
# kidease_app/api/services/job_dispatch_common.py:build_job_env) passes plain env vars instead of
# CLI flags. girderApiUrl/girderToken/input_file are gone; ITEM_ID/STORAGE_API_URL/JOB_AUTH_TOKEN/
# MODEL_ID take their place.
DEFAULT_VALS = {
    'option': 'predict',
    'white_percent': 0.01,
    'chop_thumbnail_resolution': 16,
    'overlap_percentHR': 0,
    'boxSize': 2048,
    'downsampleRateHR': 1,
    'Mag20X': False,
    'roi_thresh': 0.01,
    'min_size': [30, 30, 30, 30, 30, 30],
    'bordercrop': 300,
    'show_interstitium': True,
}


def main():
    for name, value in os.environ.items():
        print(f"{name}={value}")

    # Check if CUDA GPUs is available
    print(f"CUDAENV = {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"Allocated {torch.cuda.device_count()} cuda devices")

    args = argparse.Namespace(**DEFAULT_VALS)
    # If GPU available, or use cpu and set as args.gpu
    args.gpu = 0 if torch.cuda.device_count() > 0 else -1

    item_id = os.environ['ITEM_ID']
    storage_api_url = os.environ['STORAGE_API_URL']
    job_auth_token = os.environ['JOB_AUTH_TOKEN']
    model_id = os.environ.get('MODEL_ID')
    if not model_id:
        raise RuntimeError('MODEL_ID is required')

    client = StorageClient(storage_api_url, job_auth_token)

    mounted_path = os.getenv('TMPDIR', '/tmp')
    print(f'Downloading input for item {item_id} to {mounted_path}')
    file_path = client.download_input(item_id, mounted_path)
    print(f'Downloaded to: {file_path}')

    model_path = os.path.join(mounted_path, 'model.pth')
    print(f'Downloading model {model_id} to {model_path}')
    client.download_model(model_id, model_path)

    print(f'This is slide path: {file_path}')

    args.item_id = item_id
    args.file = file_path
    args.modelfile = model_path
    # gc.post(path=, parameters=, data=) compatible — see storage_client.StorageClient.post
    args.gc = client
    args.storage_client = client

    print(vars(args))
    for d in vars(args):
        print(f'argument: {d}, value: {getattr(args, d)}')

    run_it(args)


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
