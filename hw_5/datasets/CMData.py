from pathlib import Path
import random
import torch
import torchaudio
from torch.utils.data import Dataset


class CMData(Dataset):
    def __init__(self, part, data_dir, total_samp, limit=None, *args, **kwargs):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.part = part
        self.limit = limit
        self.total_samp = total_samp
        self.protocol = []

        if part == "train":
            add_val = "trn"
            dir_audio = self.data_dir / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof.LA.cm.{part}.{add_val}.txt"
        else:
            add_val = "trl"
            dir_audio = self.data_dir / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof.LA.cm.{part}.{add_val}.txt"

        with open(dir_audio, 'r', encoding='utf-8') as f:
            for elem in f.readlines():
                parts = elem.strip().split(" ")
                target = 0 if parts[-1] == "spoof" else 1
                self.protocol.append(
                    {"audio_path": self.data_dir / f"ASVspoof2019_LA_{part}" / "flac" / f"{parts[1]}.flac",
                     "target": target})

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(self.protocol)
            self.protocol = self.protocol[:limit]

    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, index):
        data_dict = self.protocol[index]
        audio_path = data_dict["audio_path"]
        audio_wave = self.load_audio(audio_path)
        while audio_wave.shape[-1] < self.total_samp:
            audio_wave = torch.cat((audio_wave, audio_wave), dim=-1)
        return {"audio": audio_wave[:, :self.total_samp],
                "audio_path": audio_path,
                "target": data_dict["target"]}

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = 16000
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor
