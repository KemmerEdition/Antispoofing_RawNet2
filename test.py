import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

import hw_5.model as module_model
from hw_5.trainer import Trainer
from hw_5.utils import ROOT_PATH
from hw_5.utils.object_loading import get_dataloaders
from hw_5.utils.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cpu")

    # setup data_loader instances
    # dataloaders = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    # prepare model for testing
    model = model.to(device)
    model.eval()
    test_folder = ['/content/hw_as/hw_5/test_data/audio_1.flac',
                   '/content/hw_as/hw_5/test_data/audio_2.flac',
                   '/content/hw_as/hw_5/test_data/audio_3.flac',
                   '/content/hw_as/hw_5/test_data/audio_4.flac',
                   '/content/hw_as/hw_5/test_data/audio_5.flac',
                   '/content/hw_as/hw_5/test_data/france.flac',
                   '/content/hw_as/hw_5/test_data/na.flac']

    for audio in test_folder:
        file = torchaudio.load(audio)[0].reshape(-1)
        file = file.unsqueeze(0)
        predicts = model(file)
        predict_proba =F.softmax(predicts['predicts'], dim=-1)
        # print(predict_proba)
        bonafide = predict_proba[:, 1]
        spoof = predict_proba[:, 0]
        print(f"test_file: {audio}, bonafide_proba: {int(round(bonafide[0].item()*100))}, spoof_proba: {int(round(spoof[0].item()*100))}")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    main(config, args.output)
