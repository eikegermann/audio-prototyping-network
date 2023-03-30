import os
import random
import torch


import plotly.graph_objs as go
import plotly.subplots as sp
import librosa.display
import numpy as np
import IPython.display as ipd

from .data_utils import generate_spectrogram, load_audio_file

def compute_class_prototypes(embedding_model, eval_support_loader, device):
    with torch.no_grad():
        # Calculate embeddings for the entire support set
        support_embeddings = []
        support_labels = []
        for support_set, support_label in eval_support_loader:
            support_set = support_set.to(device)
            support_embeddings.append(embedding_model(support_set))
            support_labels.extend(support_label)
        support_embeddings = torch.cat(support_embeddings, dim=0)
        support_labels = torch.tensor(support_labels, device=device)

        # Calculate class prototypes (mean embeddings)
        class_prototypes = []
        unique_labels = torch.unique(support_labels)
        for class_label in unique_labels:
            class_indices = (support_labels == class_label).nonzero(as_tuple=True)[0]
            class_embeddings = support_embeddings[class_indices]
            class_prototype = class_embeddings.mean(dim=0)
            class_prototypes.append(class_prototype)
        class_prototypes = torch.stack(class_prototypes)

    return class_prototypes

def evaluate_single_sample(audio, sr, n_fft, fixed_length, embedding_model, class_prototypes, device):
    # Preprocess the audio sample
    spec = generate_spectrogram(audio, sr, n_fft, fixed_length)

    # # Convert the test sample into a batch of size 1 and move to device
    spec = spec.unsqueeze(0).to(device)
    
    # Compute the embeddings for the input spectrogram
    input_embedding = embedding_model(spec)
    
    # Calculate the distance between the input embeddings and class prototypes
    distances = torch.cdist(input_embedding, class_prototypes)
    
    # Predict the class label based on the smallest distance
    class_probabilities = torch.softmax(-distances, dim=1)
    prediction = torch.argmax(class_probabilities, dim=1)
    
    return prediction.item()

def visualize_prediction(audio, sr, n_fft, fixed_length):
    spec = generate_spectrogram(audio, sr, n_fft, fixed_length)

    # Create a subplot with 2 rows
    fig = sp.make_subplots(rows=2, cols=1, specs=[[{'type': 'scatter'}], [{'type': 'heatmap'}]])

    # Create a waveform plot
    times = np.arange(len(audio)) / sr
    waveform_plot = go.Scatter(x=times, y=audio, mode='lines', name="Waveform")
    
    # Add the waveform plot to the subplot
    fig.add_trace(waveform_plot, row=1, col=1)

    # Create a spectrogram heatmap
    times_spec = librosa.times_like(spec.squeeze().numpy(), sr=sr)
    freqs_spec = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    spectrogram_plot = go.Heatmap(x=times_spec, y=freqs_spec, z=spec.squeeze().numpy(), colorscale='Viridis', name="Spectrogram")

    # Add the spectrogram heatmap to the subplot
    fig.add_trace(spectrogram_plot, row=2, col=1)

    # Update the layout of the subplot
    fig.update_layout(height=600, width=1000, title_text="Waveform and Spectrogram", showlegend=False)

    # Show the plot
    fig.show()

def play_audio(audio, sr):
    ipd.display(ipd.Audio(audio, rate=sr))

def pick_random_sample(test_sample_path):
    # Load test sample
    test_folders = [subfolder for subfolder in os.listdir(test_sample_path) if os.path.isdir(os.path.join(test_sample_path, subfolder))]
    test_sample_folder = random.choice(test_folders)
    test_sample_path = os.path.join(test_sample_path, test_sample_folder)
    test_samples = [file for file in os.listdir(test_sample_path) if file.endswith('.wav')]
    test_sample_file = random.choice(test_samples)
    test_sample_path = os.path.join(test_sample_path, test_sample_file)
    return test_sample_path, test_sample_folder