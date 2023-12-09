from hw_5.base.base_dataset import BaseDataset
from pathlib import Path
from hw_5.utils import ROOT_PATH


class CMDataset(BaseDataset):
    def __init__(self, part, data_dir, limit=None, *args, **kwargs):
        self.data_dir = Path(data_dir)
        self.part = part
        self.limit = limit
        index = self._parsing_protocols(part)
        super().__init__(index, *args, **kwargs)

    def _parsing_protocols(self, part):
        if part == "train":
            add_val = "trn"
            dir_audio = self.data_dir / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof.LA.cm.{part}.{add_val}.txt"
        else:
            add_val = "trl"
            dir_audio = self.data_dir / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof.LA.cm.{part}.{add_val}.txt"

        with dir_audio.open() as f:
            protocol = f.readlines()
            protocol_list = []
            for element in protocol:
                parts = element.split()
                max_sec = 4
                if max_sec <= len(parts):
                    audio_path = self.data_dir / f"ASVspoof2019_LA_{part}" / "flac" / f"{parts[1]}.flac"
                    target = 0 if parts[-1].strip() == "spoof" else 1
                    protocol_list.append({"audio_path": audio_path, "target": target})
        return protocol_list


