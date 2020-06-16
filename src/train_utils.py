import itertools

import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler


class SpeechWithNoiseDataset(Dataset):
    def __init__(self,
                 spheech_files,
                 noise_files,
                 speech_batch_size=2,
                 noise_by_speech=4,
                 max_len=None,
                 snr=(15, 10, 5, 0)):
        self.max_len = max_len
        self.speech_batch_size = speech_batch_size
        self.speech_files = list(spheech_files)
        self.noise_files = list(noise_files)
        self.noise_by_speech = noise_by_speech
        self.snr = snr

        self.speech_params = np.array(self.get_speech_params())

    def __len__(self):
        if self.max_len is None:
            return (len(self.speech_files) * len(self.speech_params) //
                    self.speech_batch_size)
        else:
            return self.max_len

    def __getitem__(self, idx):
        real_index = idx // len(self.speech_params)
        speech_data = self.__load_file(self.speech_files[real_index])

        speech_batch_index = (
            idx % len(self.speech_params)) // self.speech_batch_size
        params_range = list(
            range(speech_batch_index,
                  (speech_batch_index + self.speech_batch_size)))

        item = None
        for (noises_indexes, snr) in self.speech_params[params_range]:
            all_noise = None
            for noise_index in noises_indexes:
                noise_data = self.__load_file(self.noise_files[noise_index])
                if all_noise is None:
                    all_noise = noise_data
                else:
                    all_noise, _ = self.__add_noise(all_noise, noise_data, snr)
            speech_data_with_noise, _ = self.__add_noise(
                speech_data, all_noise, snr)

            speech_data_with_noise /= max(abs(speech_data_with_noise.min()),
                                          abs(speech_data_with_noise.max()),
                                          1e-12)

            if item is None:
                item = [
                    speech_data_with_noise.unsqueeze(0),
                    speech_data.unsqueeze(0)
                ]
            else:
                item = [
                    torch.cat((
                        prev,
                        new.unsqueeze(0),
                    ), dim=0) for prev, new in zip(
                        item,
                        [speech_data_with_noise, speech_data],
                    )
                ]
        return item

    def get_speech_params(self):
        noise_indexes = self._noise_indexes
        noise_batch = np.array_split(noise_indexes,
                                     self._noise_by_speach_count)
        iter_params = list(itertools.product(noise_batch, self.snr))
        return iter_params

    @property
    def _noise_by_speach_count(self):
        return len(self.noise_files) // self.noise_by_speech

    @property
    def _noise_indexes(self):
        return np.arange(len(self.noise_files))

    @property
    def _speech_indexes(self):
        return np.arange(len(self.speech_files))

    @staticmethod
    def __add_noise(speech, noise, snr=None):
        # Split noise
        # FIX: need len(noise) >= len(speech)
        index = np.random.randint(max(1, len(noise) - len(speech)))
        noise = noise[index:index + len(speech)]

        if snr is None:
            scale_factor = 1
        else:
            speech_mse_amplitude = torch.sqrt(torch.mean(speech**2))
            noise_mse_amplitude = torch.sqrt(torch.mean(noise**2))
            scale_factor = speech_mse_amplitude / noise_mse_amplitude / 10**(
                snr / 20)
        return speech + scale_factor * noise, noise

    def __load_file(self, path, out_sr=16000):
        data, sr = torchaudio.load(path, normalization=True)
        transform = torchaudio.transforms.Resample(
            sr, out_sr, resampling_method='sinc_interpolation')
        data = transform(data).unbind(0)[0]
        data /= max(abs(data.min()), abs(data.max()), 1e-12)
        return data
