import logging
from typing import List
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = defaultdict(list)

    for i in dataset_items:
        result_batch["audio"].append(i["audio"][0])
        result_batch["audio_path"].append(i["audio_path"])
        result_batch["target"].append(i["target"])

    for v in result_batch:
        if v == "audio":
            result_batch[v] = pad_sequence(result_batch[v], batch_first=True, padding_value=0)
        elif v == "target":
            result_batch[v] = torch.LongTensor(result_batch[v])

    return result_batch
