import os
import librosa
from source.data_utils import (number_of_filterbands, create_custom_filterbank,
                               generate_spectrogram, create_scales)



# General data parameters
sr = 44100
duration = 0.8
n_samples = int(sr * duration)

# Parameters for spectrograms
n_fft = int(4096 * 3/4)
hop_length = n_fft // 4
n_frames = librosa.time_to_frames(duration,
                                  sr=sr,
                                  hop_length=hop_length,
                                  n_fft=n_fft)
num_freqs = int(n_fft / 2 + 1)

# Parameters for custom spectrograms
fmin = 20
fmax = int(sr / 2)
fraction = 1/40
num_filters = number_of_filterbands(fmin, fmax, fraction)
filterbank = create_custom_filterbank(sr, n_fft, fraction, fmin, fmax)


# Special parameters for mel-spectrograms
n_mels = 128

# # Special parameters for wavelet spectrograms (scalograms)
lower_bound = 20
upper_bound = 20000
num_voices = 12
scales = create_scales(lower_bound, upper_bound, num_voices, sr)
fixed_length = 128

#### Set up training parameters
# Training data
data_folder = 'data/firearm_samples/train/'
fraction_for_train = 0.7
num_train_episodes = 100
support_ratio_train = 0.6

# Test data
eval_data_dir = 'data/firearm_samples/test/'
fraction_for_eval = 1
num_eval_episodes = 50
support_ratio_eval = 0.6

num_runs = 10
num_data_folders = len([subfolder for subfolder in os.listdir(data_folder)
                        if os.path.isdir(os.path.join(data_folder, subfolder))])

preprocessing_function = generate_spectrogram

num_bands = num_freqs
fixed_length = n_frames
display_interval = 10
n_features = 57
learning_rate = 6.3e-6
weight_decay = 2.3e-6

# Memory admin and balance
min_classes_per_batch = 3
batch_size_support = 18
batch_size_query = 18
min_class_appearances = 25  # Set a minimum number of appearances for each class
num_batches_train = 100
num_batches_eval = 50

# Checkpoint directory
checkpoint_dir = "checkpoints"
