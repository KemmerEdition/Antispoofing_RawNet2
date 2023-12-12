import torch
from torch import nn
from hw_5.model.RawNet2.SincConv import SincConv_fast
from hw_5.model.RawNet2.FMS_Res import ResBlock
from hw_5.base.base_model import BaseModel


class RawNet2(BaseModel):
    def __init__(self,
                 sinc_out,
                 sinc_filter,
                 res_channels_first,
                 res_channels_sec,
                 gru_units,
                 num_gru_layers,
                 num_classes):
        super().__init__()

        self.sinc = SincConv_fast(sinc_out, kernel_size=sinc_filter)
        self.max_pool = nn.MaxPool1d(3)
        self.bn1 = nn.BatchNorm1d(sinc_out)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.3)

        self.resblock1 = ResBlock(sinc_out, res_channels_first, True)
        self.resblock2 = ResBlock(res_channels_first, res_channels_first)
        self.resblock3 = ResBlock(res_channels_first, res_channels_sec)
        self.resblock4 = ResBlock(res_channels_sec, res_channels_sec)
        self.resblock5 = ResBlock(res_channels_sec, res_channels_sec)
        self.resblock6 = ResBlock(res_channels_sec, res_channels_sec)
        self.bn2 = nn.BatchNorm1d(res_channels_sec)

        self.gru = nn.GRU(res_channels_sec, gru_units, num_gru_layers, batch_first=True)
        self.liner_semi_final = nn.Linear(gru_units, gru_units)
        self.linear_final = nn.Linear(gru_units, num_classes)

    def forward(self, audio, **kwargs):
        x = audio.unsqueeze(1)
        x = torch.abs(self.sinc(x))
        x = self.leaky_relu(self.bn1(self.max_pool(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.leaky_relu(self.bn2(x))
        x = x.transpose(1, 2)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.linear_final(self.liner_semi_final(x))
        return {"predicts": x}
