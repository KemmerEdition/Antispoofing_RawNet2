import logging
import random
from typing import List

import torchaudio
from torch.utils.data import Dataset

from hw_5.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            limit=None
    ):
        self.config_parser = config_parser

        index = self._filter_records_from_dataset(index, limit)

        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["audio_path"]
        audio_wave = self.load_audio(audio_path)
        return {
            "audio": audio_wave,
            "audio_path": audio_path,
            "target": data_dict["target"]
        }

    # @staticmethod
    # def _sort_index(index):
    #     return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    @staticmethod
    def _filter_records_from_dataset(
            index: list, limit
    ) -> list:

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index
