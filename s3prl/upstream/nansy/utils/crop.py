import torch
from .mel import load_mel_from_audio

class CropUtil:

    def __init__(self) -> None:
        self.w_len = 2048  # for yingram
        self.mel_window = 128  # window size in mel
        self.hop_size = 256
        self.audio_window_22k = self.mel_window * self.hop_size  # 32768, window size in raw audio
        self.segment_duration = self.audio_window_22k / 22050  # 1.486 (sec)
        self.audio_window_16k = int(self.segment_duration * 16000)  # 23778

        self.yin_window_22k = self.audio_window_22k + self.w_len  # 34816
        self.yin_segment_duration = self.yin_window_22k / 22050.  # 1.579

        zero_audio = torch.zeros(self.yin_window_22k).float()
        zero_mel = load_mel_from_audio(zero_audio)
        self.mel_padding_value = torch.min(zero_mel).data

        self.minimum_audio_length = self.yin_window_22k
        self.minimum_mel_length = zero_mel.shape[-1]

    @staticmethod
    def pad_audio(x: torch.Tensor, length: int, value: float = 0., pad_at: str = 'end') -> torch.Tensor:
        r"""pads value to audio data, at last dimension

        params:
            x: torch.Tensor of shape (..., T)
            length: int, length to pad
            value: float, value to pad
            pad_at: str, 'start' or 'end'

        returns:
            y: padded torch.Tensor of shape (..., T+length)
        """
        # x: (..., T)
        pad_at = pad_at.strip().lower()
        if pad_at == 'end':
            y = torch.cat([
                x, torch.ones(*x.shape[:-1], length) * value
            ], dim=-1)
        elif pad_at == 'start':
            y = torch.cat([
                torch.ones(*x.shape[:-1], length) * value, x
            ], dim=-1)
        else:
            raise NotImplementedError
        return y

    @staticmethod
    def crop_tensor(x: torch.Tensor, start: int, end: int, padding_value: float = 0.) -> torch.Tensor:
        r"""crop tensor at last dimension from start to end, automatically pad with padding_value

        params:
            x: torch.Tensor of shape (..., T)
            start: int, position to crop
            end: int, position to crop
            padding_value: float, value for padding when needed

        returns:
            y: torch.Tensor of shape (..., end-start)
        """
        if start < 0:
            if end < 0:
                y = torch.ones(size=(*x.shape[:-1], end - start), dtype=torch.float, device=x.device) * padding_value
            elif end > x.shape[-1]:
                y = x
                y = CropUtil.pad_audio(y, -start, padding_value, pad_at='start')
                y = CropUtil.pad_audio(y, end - x.shape[-1], padding_value, pad_at='end')
            else:
                y = x[..., :end]
                y = CropUtil.pad_audio(y, -start, padding_value, pad_at='start')
        elif end > x.shape[-1]:
            if start > x.shape[-1]:
                y = torch.ones(size=(*x.shape[:-1], end - start), dtype=torch.float, device=x.device) * padding_value
            else:
                y = x[..., start:]
                y = CropUtil.pad_audio(y, end - x.shape[-1], padding_value, pad_at='end')
        else:
            y = x[..., start:end]
        assert y.shape[-1] == end - start, f'{x.shape}, {start}, {end}, {y.shape}'
        return y

    def get_time_idxs(self, mel_start: int):
        r"""calculates time-related idxs needed for getitem

        params:
            mel_start: idx where splitted mel starts

        returns:
            mel_start: int, start index of mel
            mel_end: int, end index of mel
            t_start: float, start time (sec)
            w_start_16k: int, start index of 16k audio
            w_start_22k: int, start index of 22k audio
            w_end_16k: int, end index of 16k audio
            w_end_22k: int, end index of 22k audio
            w_end_22k_yin: int, end index of 22k audio for yin computation
        """
        mel_end = mel_start + self.mel_window

        t_start = mel_start * self.hop_size / 22050.
        w_start_22k = int(t_start * 22050)
        w_start_16k = int(t_start * 16000)
        w_end_22k = w_start_22k + self.audio_window_22k
        w_end_22k_yin = w_start_22k + self.yin_window_22k
        w_end_16k = w_start_16k + self.audio_window_16k

        return mel_start, mel_end, t_start, w_start_16k, w_start_22k, w_end_16k, w_end_22k, w_end_22k_yin