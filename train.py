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

from source.dataset import CustomDataset
from source.architecture import AudioClassifier
from source.data_utils import generate_spectrogram
import config as conf


# Run multiple training runs to find best checkpoint

# Get number of classes


train_dataset = CustomDataset(conf.data_folder,
                              conf.preprocessing_function,
                              conf.sr,
                              conf.duration,
                              conf.n_fft,
                              conf.fixed_length)

test_dataset = CustomDataset(conf.eval_data_dir,
                              conf.preprocessing_function,
                              conf.sr,
                              conf.duration,
                              conf.n_fft,
                              conf.fixed_length)
                             

# Set variable to compare evaluation accuracies
best_accuracy = 0.0
best_f1_score = 0.0

# Create a directory to save the best checkpoint
os.makedirs(conf.checkpoint_dir, exist_ok=True)

for model_run in range(conf.num_runs):
    print(f"Model run: {model_run}")
    #### Training Phase
    # Instantiate the embedding model
    embedding_model = AudioClassifier(n_bands=conf.num_bands,
                                      n_frames=conf.fixed_length,
                                      n_features=conf.n_features)

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(embedding_model.parameters(),
                          lr=conf.learning_rate,
                          weight_decay=conf.weight_decay)
    
    # # Create the CosineAnnealingLR scheduler
    # T_max = 45  # You can choose an appropriate value for T_max
    # eta_min = 5e-7  # You can choose an appropriate value for eta_min
    # scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model = embedding_model.to(device)

    # Set up interval display for high numbers of episodes
    total_loss = 0
    total_accuracy = 0
    all_true_labels = []
    all_predictions = []

    class_counter = {class_label: 0 for class_label in range(conf.num_data_folders)}
    all_classes_represented = all(count >= conf.min_class_appearances for count in class_counter.values())
    step_num = 0

    # Training loop
    for episode in range(conf.num_episodes):

        train_support_loader, train_query_loader = train_dataset.get_support_query_dataloaders(conf.min_classes_per_batch,
                                                                                               conf.batch_size_support,
                                                                                               conf.batch_size_query,
                                                                                               conf.num_batches)

        eval_support_loader, eval_query_loader = test_dataset.get_support_query_dataloaders(conf.min_classes_per_batch,
                                                                                            conf.batch_size_support,
                                                                                            conf.batch_size_query,
                                                                                            conf.num_batches)

        while step_num < conf.num_batches:
            # Load one batch from support and query DataLoaders
            print("Establishing support set... ")
            support_set, support_labels = next(iter(train_support_loader))
            print("Establishing query set... ")
            query_set, query_labels = next(iter(train_query_loader))

            # Load one batch from support and query DataLoaders
            
            support_set, support_labels = support_set.to(device), support_labels.to(device)
            
            query_set, query_labels = query_set.to(device), query_labels.to(device)

            batch_labels = torch.unique(support_labels)
            print("Support batch labels: ", batch_labels)
            print("Query batch labels: ", torch.unique(query_labels))

            # Calculate embeddings for support and query sets
            support_embeddings = embedding_model(support_set)
            query_embeddings = embedding_model(query_set)

            # Calculate class prototypes (mean embeddings)
            class_prototypes = []
            for class_label in batch_labels:
                class_indices = (support_labels == class_label).nonzero(as_tuple=True)[0]
                class_embeddings = support_embeddings[class_indices]
                class_prototype = class_embeddings.mean(dim=0)
                class_prototypes.append(class_prototype)
            class_prototypes = torch.stack(class_prototypes)

            # Calculate the distance between query samples and class prototypes
            distances = torch.cdist(query_embeddings, class_prototypes)

            # Predict the class label based on the smallest distance
            class_probabilities = torch.softmax(-distances, dim=1)
            print(query_labels)
            # Remap query labels to the new range
            query_labels_remap = torch.tensor([torch.where(batch_labels == label)[0].item() for label in query_labels], device=device)

            # Calculate the loss and optimize the model
            loss = criterion(class_probabilities, query_labels_remap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Update the counter for each selected class
            for class_label in batch_labels:
                class_counter[class_label.item()] += 1

            # Check if all classes have been represented the minimum number of times
            all_classes_represented = all(count >= conf.min_class_appearances for count in class_counter.values())

            #increase counter
            step_num += 1

            # Calculate accuracy and loss for the current episode
            predictions = torch.argmax(class_probabilities, dim=1)
            accuracy = (predictions == query_labels_remap).float().mean().item()
            total_loss += loss.item()
            total_accuracy += accuracy

            all_true_labels.extend(query_labels_remap.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

        if (episode + 1) % conf.display_interval == 0:
            avg_loss = total_loss / conf.display_interval
            avg_accuracy = total_accuracy / conf.display_interval
            f1 = f1_score(all_true_labels, all_predictions, average='weighted')
            
            print(f"Episode {episode + 1}/{conf.num_episodes}, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}, F1 Score: {f1:.4f}")

            total_loss = 0
            total_accuracy = 0
            all_true_labels.clear()
            all_predictions.clear()

    #### Evaluation phase
    embedding_model.eval()  # Set the model to evaluation mode

    # Set up evaluation accuracy display
    eval_total_accuracy = 0
    eval_total_loss = 0
    all_eval_true_labels = []
    all_eval_predictions = []

    # Evaluation loop
    for episode in range(conf.num_eval_episodes):
        # Randomly select a subset of classes
        available_classes = torch.unique(eval_labels_tensor)
        batch_labels = available_classes[torch.randperm(len(available_classes))[:conf.num_batch_labels_eval]]
        batch_labels = batch_labels.to(device)

        # Initialize support and query sets
        eval_support_set = torch.empty((0, 1, num_bands, fixed_length))
        eval_support_labels = torch.empty(0, dtype=torch.long)
        eval_query_set = torch.empty((0, 1, num_bands, fixed_length))
        eval_query_labels = torch.empty(0, dtype=torch.long)

        # Split the data into support and query sets
        for class_label in batch_labels:
            class_indices = [i for i, label in enumerate(eval_labels) if label == class_label]
            random.shuffle(class_indices)
            n_support = int(conf.support_ratio_eval * len(class_indices))
            support_indices = class_indices[:n_support]
            query_indices = class_indices[n_support:]

            eval_support_set = torch.cat((eval_support_set, eval_data[support_indices]), dim=0)
            eval_support_labels = torch.cat((eval_support_labels, eval_labels_tensor[support_indices]), dim=0)
            eval_query_set = torch.cat((eval_query_set, eval_data[query_indices]), dim=0)
            eval_query_labels = torch.cat((eval_query_labels, eval_labels_tensor[query_indices]), dim=0)

        # Move data to available device
        eval_support_set = eval_support_set.to(device)
        eval_support_labels = eval_support_labels.to(device)
        eval_query_set = eval_query_set.to(device)
        eval_query_labels = eval_query_labels.to(device)

        # Calculate embeddings for support and query sets
        eval_support_embeddings = embedding_model(eval_support_set)
        eval_query_embeddings = embedding_model(eval_query_set)

        # Calculate class prototypes (mean embeddings)
        class_prototypes = []
        for class_label in batch_labels:
            class_indices = (eval_support_labels == class_label).nonzero(as_tuple=True)[0]
            class_embeddings = eval_support_embeddings[class_indices]
            class_prototype = class_embeddings.mean(dim=0)
            class_prototypes.append(class_prototype)
        class_prototypes = torch.stack(class_prototypes)

        # Calculate the distance between query samples and class prototypes
        distances = torch.cdist(eval_query_embeddings, class_prototypes)

        # Predict the class label based on the smallest distance
        class_probabilities = torch.softmax(-distances, dim=1)

        # Remap query labels to the new range
        eval_query_labels_remap = torch.tensor([torch.where(batch_labels == label)[0].item() for label in eval_query_labels], device=device)

        # Calculate the loss for the current episode
        loss = criterion(class_probabilities, eval_query_labels_remap)

        # Calculate accuracy for the current episode
        predictions = torch.argmax(class_probabilities, dim=1)
        accuracy = (predictions == eval_query_labels_remap).float().mean().item()
        eval_total_loss += loss.item()
        eval_total_accuracy += accuracy

        all_eval_true_labels.extend(eval_query_labels_remap.cpu().numpy())
        all_eval_predictions.extend(predictions.cpu().numpy())

    # Calculate evaluation metrics
    eval_avg_loss = eval_total_loss / conf.num_eval_episodes
    eval_avg_accuracy = eval_total_accuracy / conf.num_eval_episodes
    f1 = f1_score(all_eval_true_labels, all_eval_predictions, average='weighted')

    print(f"Evaluation Average Loss: {eval_avg_loss:.4f}, Average Accuracy: {eval_avg_accuracy:.4f}, F1 Score: {f1:.4f} \n")

    if f1 > best_f1_score:
        print(f"Previous best F1 score: {best_f1_score:.4f} - current F1 score: {f1:.4f}")
        best_f1_score = f1
        best_model_weights = copy.deepcopy(embedding_model.state_dict())
        print("Saving new best checkpoint... \n")
        torch.save(best_model_weights, os.path.join(conf.checkpoint_dir, 'best_checkpoint.pth'))

print(f"Evaluation complete. Saved checkpoint has F1 score of {best_f1_score:.2f}")
