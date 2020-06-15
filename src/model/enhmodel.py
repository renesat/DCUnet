import torch
import torch.nn as nn
from torchaudio.functional import istft


class EnhModel(nn.Module):
    def __init__(self, conv, n_fft, hop_len):
        super(EnhModel, self).__init__()

        self.n_fft = n_fft
        self.hop_len = hop_len
        self.window = torch.hann_window(n_fft)
        self.conv = conv

    def forward(self, audio):
        x_stft = torch.stft(audio,
                            self.n_fft,
                            self.hop_len,
                            window=self.window)  # (B, W, H, 2)
        x_conv = self.conv(x_stft.unsqueeze(1)).unbind(1)[0]  # (B, W, H, 2)
        x_crm = self.cRM(x_conv, x_stft)
        x_istft = istft(x_crm, self.n_fft, self.hop_len, window=self.window)
        return x_istft

    def cRM(self, out_x, in_x):
        mask = torch.sqrt(out_x.unbind(3)[0]**2 + out_x.unbind(3)[1]**2)
        mask = torch.tanh(mask)

        data = torch.sqrt(in_x.unbind(3)[0]**2 + in_x.unbind(3)[1]**2)
        data *= mask

        angle = torch.atan2(*out_x.unbind(3))
        angle += torch.atan2(*in_x.unbind(3))

        x = torch.stack((
            data * torch.cos(angle),
            data * torch.sin(angle),
        ),
                        dim=3)
        return x
