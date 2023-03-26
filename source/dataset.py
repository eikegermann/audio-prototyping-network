import os
import librosa
import librosa.display
import itertools
import numpy as np
import IPython.display as ipd
from torch.utils.data import Dataset, DataLoader

class EpisodicBatchSampler():
    def __init__(
        self, labels, num_batches, min_classes_per_batch, samples_per_class, num_random_classes=1
    ):
        self.labels = labels
        self.num_batches = num_batches
        self.min_classes_per_batch = min_classes_per_batch
        self.samples_per_class = samples_per_class
        self.num_random_classes = num_random_classes

        self.num_classes = len(np.unique(labels))
        self.class_cycle = itertools.cycle(range(self.num_classes))

    def __iter__(self):
        for _ in range(self.num_batches):
            # Sample the first classes using the round-robin approach
            sampled_classes = [next(self.class_cycle) for _ in range(self.min_classes_per_batch - self.num_random_classes)]

            # Sample the additional random classes
            remaining_classes = list(set(range(self.num_classes)) - set(sampled_classes))
            random_classes = np.random.choice(remaining_classes, self.num_random_classes, replace=False)
            sampled_classes.extend(random_classes)

            # Ensure sampled_classes contains unique elements
            while len(set(sampled_classes)) != self.min_classes_per_batch:
                sampled_classes[-1] = next(self.class_cycle)

            batch_indices = []

            for class_label in sampled_classes:
                class_indices = np.where(np.array(self.labels) == class_label)[0]
                np.random.shuffle(class_indices)
                batch_indices.extend(class_indices[: self.samples_per_class])

            np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches



class CustomDataset(Dataset):
    def __init__(self, data_folder, preprocessing_function, sr, duration, n_fft, fixed_length):
        self.data, self.labels = self.load_audio_files(data_folder, sr, duration)
        self.preprocessing_function = preprocessing_function
        self.sr = sr
        self.n_fft = n_fft
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio = self.data[idx]
        spectrogram = self.preprocessing_function(audio, self.sr, self.n_fft, self.fixed_length)
        label = self.labels[idx]
        return spectrogram, label

    def load_audio_files(self, data_folder, sr, duration):
        class_folders = [subfolder for subfolder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, subfolder))]
        data = []
        labels = []

        for class_label, class_folder in enumerate(class_folders):
            class_path = os.path.join(data_folder, class_folder)
            audio_files = [file for file in os.listdir(class_path) if file.endswith('.wav')]
            for audio_file in audio_files:
                file_path = os.path.join(class_path, audio_file)
                audio = self.load_audio_file(file_path, sr, duration)
                data.append(audio)
                labels.append(class_label)

        return data, labels

    def load_audio_file(self, file_path, sr, duration):
        audio, _ = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast')
        if len(audio) < int(sr * duration):
            audio = np.pad(audio, (0, int(sr * duration) - len(audio)), mode='constant')
        return audio
    
    def get_support_query_dataloaders(self, min_classes_per_batch, batch_size_support, batch_size_query, num_batches_per_episode):
        num_classes = len(np.unique(self.labels))
        if min_classes_per_batch > num_classes:
            raise ValueError("min_classes_per_batch should be less than or equal to the number of unique classes in the dataset")
        
        samples_per_class_support = batch_size_support // min_classes_per_batch
        samples_per_class_query = batch_size_query // min_classes_per_batch

        support_sampler = EpisodicBatchSampler(self.labels, num_batches_per_episode, min_classes_per_batch, samples_per_class_support)
        query_sampler = EpisodicBatchSampler(self.labels, num_batches_per_episode, min_classes_per_batch, samples_per_class_query)

        support_dataloader = DataLoader(self, batch_sampler=support_sampler)
        query_dataloader = DataLoader(self, batch_sampler=query_sampler)

        return support_dataloader, query_dataloader
