import os
import librosa
import librosa.display
import pywt
import numpy as np
import torch




## Base data functions

def load_audio_file(file_path, sr, duration):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast')
    if len(audio) < int(sr * duration):
        audio = np.pad(audio, (0, int(sr * duration) - len(audio)), mode='constant')
    return audio


def load_audio_files(data_folder, sr, duration):
    class_folders = [subfolder for subfolder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, subfolder))]
    data = []
    labels = []

    for class_label, class_folder in enumerate(class_folders):
        class_path = os.path.join(data_folder, class_folder)
        audio_files = [file for file in os.listdir(class_path) if file.endswith('.wav')]
        for audio_file in audio_files:
            file_path = os.path.join(class_path, audio_file)
            audio = load_audio_file(file_path, sr, duration)
            data.append(audio)
            labels.append(class_label)

    return data, labels, class_folders


def pad_or_truncate(tensor, fixed_length):
    if tensor.size(-1) < fixed_length:
        return torch.nn.functional.pad(tensor, (0, fixed_length - tensor.size(-1)))
    else:
        return tensor[..., :fixed_length]


## Mel-spectrograms

def generate_mel_spectrogram(audio, sr, n_fft, n_mels, fixed_length):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)
    mel_spec = pad_or_truncate(mel_spec, fixed_length)
    return mel_spec


def generate_mel_spectrograms(data, sr, n_fft, n_mels, fixed_length):
    mel_spectrograms = []
    for audio in data:
        mel_spec = generate_mel_spectrogram(audio, sr, n_fft, n_mels, fixed_length)
        mel_spectrograms.append(mel_spec)
    return torch.stack(mel_spectrograms)


## General STFT-spectrograms
##### TODO: Use normalisation for network, db for display
##### TODO: Try to include phase information by not squaring values

def generate_spectrogram(audio, sr, n_fft, fixed_length):
    spec = librosa.stft(y=audio, n_fft=n_fft)
    spec = np.abs(spec) ** 2
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = torch.FloatTensor(spec).unsqueeze(0)
    spec = pad_or_truncate(spec, fixed_length)
    return spec


def generate_spectrograms(data, sr, n_fft, fixed_length):
    spectrograms = []
    for audio in data:
        spec = generate_spectrogram(audio, sr, n_fft, fixed_length)
        spectrograms.append(spec)
    return torch.stack(spectrograms)


## Custom spectrograms

def number_of_filterbands(fmin, fmax, fraction):
    n = 1 + np.ceil(np.log2((fmax/fmin)**(1/fraction)))
    return int(n)


def create_custom_filterbank(sr, n_fft, fraction, fmin, fmax):
    # Calculate the center frequencies for the given fractional octave band
    num_bands = number_of_filterbands(fmin, fmax, fraction)
    center_frequencies_hz = np.geomspace(fmin, fmax, num=num_bands)

    # Calculate the bandwidths for each filter
    bandwidths_hz = center_frequencies_hz[1:] - center_frequencies_hz[:-1]

    # Calculate corresponding FFT bins
    fft_bins = np.floor((n_fft + 1) * center_frequencies_hz / sr).astype(int)

    # Create the filterbank matrix
    filterbank = np.zeros((len(center_frequencies_hz), int(n_fft // 2 + 1)))

    # Construct filters
    for i in range(len(center_frequencies_hz) - 1):
        center = fft_bins[i]
        half_bandwidth = int(round(bandwidths_hz[i] * n_fft / sr))

        left = center - half_bandwidth
        right = center + half_bandwidth

        filterbank[i, left:center] = (np.arange(left, center) - left) / (center - left)
        filterbank[i, center:right] = (right - np.arange(center, right)) / (right - center)

    return filterbank

def generate_custom_spectrogram(audio, filterbank, n_fft, fixed_length):
    spec = librosa.stft(y=audio, n_fft=n_fft)
    spec = np.abs(spec) ** 2
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = np.dot(filterbank, spec)
    spec = torch.FloatTensor(spec).unsqueeze(0)
    spec = pad_or_truncate(spec, fixed_length)
    return spec


def generate_custom_spectrograms(data, filterbank, n_fft, fixed_length):
    spectrograms = []
    for audio in data:
        spec = generate_custom_spectrogram(audio, filterbank,
                                           sr, n_fft, fixed_length)
        spectrograms.append(spec)
    return torch.stack(spectrograms)



## Wavelet spectrograms

def generate_cwt_scalogram(audio, sr, scales, fixed_length):
    coef, _ = pywt.cwt(audio, scales, 'morl', 1 / sr)
    coef = np.abs(coef) ** 2
    coef = 10 * np.log10(coef + np.finfo(float).eps)
    coef = torch.FloatTensor(coef).unsqueeze(0)
    coef = pad_or_truncate(coef, fixed_length)
    return coef

def generate_cwt_scalograms(data, sr, scales, fixed_length):
    scalograms = []
    for audio in data:
        scalogram = generate_cwt_scalogram(audio, sr, scales, fixed_length)
        scalograms.append(scalogram)
    return torch.stack(scalograms)

def create_scales(f_min, f_max, voices_per_octave, sr):
    num_octaves = np.log2(f_max / f_min)
    num_scales = int(num_octaves * voices_per_octave)
    return np.logspace(np.log2(f_min * sr), np.log2(f_max * sr), num_scales, base=2)
