import random
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torchaudio import functional as AF

from typing import List, Dict, Union
from collections import OrderedDict

from .modules.analysis import Analysis
from .utils.crop import CropUtil
from .utils.mel import load_mel_from_audio

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, **kwargs):
        super().__init__()
        models = torch.load(ckpt)
        self.model = Analysis()
        component_state_dict = OrderedDict()
        for key in models.keys():
            if 'network.Analysis.' in key:
                component_state_dict[key.replace('network.Analysis.', '')] = models[key]
        self.model.load_state_dict(component_state_dict, strict=False)
        self.crop_util = CropUtil()

    def get_downsample_rates(self, key: str) -> int:
        return 1

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        Args:
            wavs:
                list of unpadded wavs [wav1, wav2, ...]
                each wav is in torch.FloatTensor with sample rate 16000
                and already put in the device assigned by command-line args
        Return:
            hidden states:
                list of analysis results from the NANSY analysis module, including:
                - Linguistic feature
                - Speaker feature
                - Energy feature
                - Pitch feature
                please refer to section 2.1 of https://arxiv.org/abs/2110.14513 for their structure and usage
        """

        wavs_16k = wavs
        wavs_22k = [AF.resample(wav, 16000, 22050) for wav in wavs]
        mels_22k = [load_mel_from_audio(wav) for wav in wavs_22k]

        if self.training:
            # train
            cropped_wavs_16k = []
            cropped_wavs_22k_yin = []
            cropped_mels_22k = []

            for wav_16k, wav_22k, mel in zip(wavs_16k, wavs_22k, mels_22k):
                mel_start = random.randint(0, mel.shape[-1] - self.crop_util.minimum_mel_length)
                time_idxs = self.crop_util.get_time_idxs(mel_start)
                cropped_wavs_16k.append(CropUtil.crop_tensor(wav_16k, time_idxs[3], time_idxs[5]))
                cropped_wavs_22k_yin.append(CropUtil.crop_tensor(wav_22k, time_idxs[4], time_idxs[7]))
                cropped_mels_22k.append(CropUtil.crop_tensor(mel, time_idxs[0], time_idxs[1],
                                        padding_value=self.crop_util.mel_padding_value))
                
            batch = {
                'wavs_16k': pad_sequence(cropped_wavs_16k, batch_first=True),
                'wavs_22k_yin': pad_sequence(cropped_wavs_22k_yin, batch_first=True),
                'mels_22k': pad_sequence(cropped_mels_22k, batch_first=True)
            }
            
            with torch.no_grad():
                linguistic_feat = self.model.linguistic(batch['wavs_16k'])
                energy_feat = self.model.energy(batch['mels_22k'])
                pitch_feat = self.model.pitch.yingram_batch(batch['wavs_22k_yin'])

            speaker_feat = self.model.speaker(batch['wavs_16k'])
            
        else:
            # eval
            pass

        return {
            "hidden_states": [
                linguistic_feat,
                speaker_feat,
                energy_feat,
                pitch_feat,
            ],
        }
