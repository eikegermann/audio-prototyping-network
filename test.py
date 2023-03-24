import os
import copy
import math
import random
import librosa
import librosa.display
import itertools
import pywt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import IPython.display as ipd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from IPython.display import Audio, display
from torchsummary import summary

# def load_single_sample(file_path, sr, duration, n_fft, n_mels, n_frames):
#     audio = load_audio_file(file_path, sr, duration)
#     mel_spectrogram = generate_mel_spectrogram(audio, sr, n_fft, n_mels, n_frames)
#     return audio, mel_spectrogram

# def visualize_prediction(audio, S, true_class, predicted_class):
#     fig, ax = plt.subplots(2, 1, figsize=(10, 6))

#     # Plot the waveform
#     ax[0].set_title("Waveform")
#     librosa.display.waveshow(audio, sr=sr, ax=ax[0])

#     # Plot the spectrogram
#     img = librosa.display.specshow(S.squeeze().numpy(), x_axis='time', y_axis='mel', sr=sr, fmax=sr // 2, ax=ax[1])
#     ax[1].set_title("Mel Spectrogram")
#     fig.colorbar(img, ax=ax[1], format="%+2.f dB")

#     plt.show()

#     print("True Class Label:", true_class)
#     print("Predicted Class Label:", predicted_class)

# def play_audio(audio):
#     ipd.display(ipd.Audio(audio, rate=sr))

# test_sample_path = 'drive/MyDrive/audio_ml_data/firearm_samples/test/'

# # Load test sample
# test_folders = [subfolder for subfolder in os.listdir(test_sample_path) if os.path.isdir(os.path.join(test_sample_path, subfolder))]
# test_sample_folder = random.choice(test_folders)
# test_sample_path = os.path.join(test_sample_path, test_sample_folder)
# test_samples = [file for file in os.listdir(test_sample_path) if file.endswith('.wav')]
# test_sample_file = random.choice(test_samples)
# test_sample_path = os.path.join(test_sample_path, test_sample_file)

# audio, test_sample = load_single_sample(test_sample_path, sr, duration, n_fft, n_mels, n_frames)

# # Convert the test sample into a batch of size 1
# test_sample = test_sample.unsqueeze(0).to(device)

# # Load the best checkpoint
# embedding_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_checkpoint.pth')))
# embedding_model.eval()

# # Support set data
# support_data_folder = 'drive/MyDrive/audio_ml_data/firearm_samples/train/'

# # Load audio files and labels
# support_data, support_labels = load_audio_files(support_data_folder, sr, duration)

# # Generate mel-spectrograms
# support_data = generate_mel_spectrograms(support_data, sr, n_fft, n_mels, n_frames)
# support_labels_tensor = torch.tensor(support_labels)

# # Calculate embeddings for support and test samples
# support_embeddings = embedding_model(support_data.to(device))
# test_sample_embedding = embedding_model(test_sample)

# # Calculate class prototypes (mean embeddings)
# class_prototypes = []
# available_classes = torch.unique(support_labels_tensor)
# for class_label in available_classes:
#     class_indices = (support_labels_tensor == class_label).nonzero(as_tuple=True)[0]
#     class_embeddings = support_embeddings[class_indices]
#     class_prototype = class_embeddings.mean(dim=0)
#     class_prototypes.append(class_prototype)
# class_prototypes = torch.stack(class_prototypes)

# # Calculate the distance between the test sample and class prototypes
# distances = torch.cdist(test_sample_embedding, class_prototypes)

# # Predict the class label based on the smallest distance
# class_probabilities = torch.softmax(-distances, dim=1)
# prediction = torch.argmax(class_probabilities, dim=1)

# # Move the prediction tensor to the CPU
# prediction = prediction.cpu()

# # Map the predicted class label back to the original class name
# predicted_class = available_classes[prediction]

# print(f"Classes: {test_folders}")
# print(f"Predicted class for the test sample '{test_sample_file}' is: {test_folders[predicted_class.item()]}")

# # Make information available for visualisation
# true_class = test_sample_folder
# pred_class = test_folders[predicted_class.item()]
# sample = test_sample.cpu()

# # Visualize the prediction
# visualize_prediction(audio, sample, true_class, pred_class)

# # Play the audio sample
# play_audio(audio)
