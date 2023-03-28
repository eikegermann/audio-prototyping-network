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

from torch.nn.functional import pairwise_distance
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, fbeta_score, roc_auc_score, roc_curve

from IPython.display import Audio, display
from torchsummary import summary

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from source.dataset import CustomDataset
from source.architecture import AudioClassifier
from source.data_utils import generate_spectrogram
import config as conf

def train_pt_classifier(conf, train_dataset, test_dataset):
    print(f"Beginning training...")
    #### Training Phase
    # Instantiate the embedding model
    embedding_model = AudioClassifier(n_bands=conf.num_bands,
                                      n_frames=conf.fixed_length,
                                      n_features=conf.n_features)

    # Set up the loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    margin = 1.0
    optimizer = optim.Adam(embedding_model.parameters(),
                          lr=conf.learning_rate,
                          weight_decay=conf.weight_decay)
    
    # # Create the CosineAnnealingLR scheduler
    # T_max = 18  # You can choose an appropriate value for T_max
    # eta_min = 5e-7  # You can choose an appropriate value for eta_min
    # scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model = embedding_model.to(device)

    # Set up interval display for high numbers of episodes
    total_loss = 0
    total_accuracy = 0
    all_true_labels = []
    all_predictions = []



    # Training loop
    for episode in range(conf.num_train_episodes):
        class_counter = {class_label: 0 for class_label in range(conf.num_data_folders)}
        all_classes_represented = all(count >= conf.min_class_appearances for count in class_counter.values())
        step_num = 0

        train_support_loader, train_query_loader = train_dataset.get_support_query_dataloaders(conf.min_classes_per_batch,
                                                                                               conf.batch_size_support,
                                                                                               conf.batch_size_query,
                                                                                               conf.num_batches_train)

        while ((step_num < conf.num_batches_train) and not(all_classes_represented)):
            # print(all_classes_represented)
            # Load one batch from support and query DataLoaders
            # print("Establishing support set... ")
            support_set, support_labels = next(iter(train_support_loader))
            # print("Establishing query set... ")
            query_set, query_labels = next(iter(train_query_loader))

            # Load one batch from support and query DataLoaders
            
            support_set, support_labels = support_set.to(device), support_labels.to(device)
            
            query_set, query_labels = query_set.to(device), query_labels.to(device)

            batch_labels = torch.unique(support_labels)
            # print("Support batch labels: ", batch_labels)
            # print("Query batch labels: ", torch.unique(query_labels))

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
            # Remap query labels to the new range
            query_labels_remap = torch.tensor([torch.where(batch_labels == label)[0].item() for label in query_labels], device=device)

            # Find the indices of the positive and negative examples
            positive_indices = torch.argmin(distances, dim=1)
            negative_indices = torch.argmin(distances + torch.eye(distances.size(0), distances.size(1)).to(device) * 1e9, dim=1)

            # Get the embeddings for anchors, positives, and negatives
            anchor_embeddings = class_prototypes[query_labels_remap]
            positive_embeddings = query_embeddings
            negative_embeddings = class_prototypes[negative_indices]

            # Calculate the pairwise distances for positive and negative pairs
            positive_distances = pairwise_distance(anchor_embeddings, positive_embeddings)
            negative_distances = pairwise_distance(anchor_embeddings, negative_embeddings)

            # Calculate the contrastive loss
            loss = (positive_distances ** 2 + torch.clamp(margin - negative_distances, min=0) ** 2).mean() / 2

            # Optimize the model
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
            # auc_roc = roc_auc_score(query_labels_remap, class_probabilities)
            predictions = torch.argmax(class_probabilities, dim=1)
            accuracy = (predictions == query_labels_remap).float().mean().item()
            # fbeta = fbeta_score(query_labels_remap, predictions, beta=0.9)
            total_loss += loss.item()
            total_accuracy += accuracy

            all_true_labels.extend(query_labels_remap.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

        # print(class_counter)
        if (episode + 1) % conf.display_interval == 0:
            avg_loss = (total_loss / step_num) / conf.display_interval
            avg_accuracy = (total_accuracy / step_num) / conf.display_interval
            f1 = f1_score(all_true_labels, all_predictions, average='weighted')
            f_beta = fbeta_score(all_true_labels, all_predictions, beta=0.9, average='weighted')
            
            print(f"Episode {episode + 1}/{conf.num_train_episodes}, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}, F1 Score: {f1:.4f}, F-beta: {f_beta:.4f}")

            total_loss = 0
            total_accuracy = 0
            all_true_labels.clear()
            all_predictions.clear()

    #### Evaluation phase
    print("Beginning evaluation... ")
    best_accuracy = 0.0
    best_f1_score = 0.0
    best_f_beta_score = 0.0
    embedding_model.eval()  # Set the model to evaluation mode

    # Set up evaluation accuracy display
    eval_total_accuracy = 0
    eval_total_loss = 0
    all_eval_true_labels = []
    all_eval_predictions = []

    # Evaluation loop
    for episode in range(conf.num_eval_episodes):
        # print(f"Running episode {episode} of {conf.num_eval_episodes}")
        eval_support_loader, eval_query_loader = test_dataset.get_support_query_dataloaders(conf.min_classes_per_batch,
                                                                                            conf.batch_size_support,
                                                                                            conf.batch_size_query,
                                                                                            conf.num_batches_eval)
        step_num = 0
        while step_num < conf.num_batches_eval:
            # print(f"Running eval step {step_num} of {conf.num_batches_eval}")
            # Load one batch from support and query DataLoaders
            # print("Establishing support set... ")
            eval_support_set, eval_support_labels = next(iter(eval_support_loader))
            # print("Establishing query set... ")
            eval_query_set, eval_query_labels = next(iter(eval_query_loader))

            # Load one batch from support and query DataLoaders
            
            eval_support_set, eval_support_labels = eval_support_set.to(device), eval_support_labels.to(device)
            
            eval_query_set, eval_query_labels = eval_query_set.to(device), eval_query_labels.to(device)

            batch_labels = torch.unique(eval_support_labels)
            # print("Support batch labels: ", batch_labels)
            # print("Query batch labels: ", torch.unique(eval_query_labels))

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

            # Find the indices of the positive and negative examples
            positive_indices = torch.argmin(distances, dim=1)
            negative_indices = torch.argmin(distances + torch.eye(distances.size(0), distances.size(1)).to(device) * 1e9, dim=1)

            # Get the embeddings for anchors, positives, and negatives
            anchor_embeddings = class_prototypes[query_labels_remap]
            positive_embeddings = query_embeddings
            negative_embeddings = class_prototypes[negative_indices]

            # Calculate the pairwise distances for positive and negative pairs
            positive_distances = pairwise_distance(anchor_embeddings, positive_embeddings)
            negative_distances = pairwise_distance(anchor_embeddings, negative_embeddings)

            # Calculate the contrastive loss
            loss = (positive_distances ** 2 + torch.clamp(margin - negative_distances, min=0) ** 2).mean() / 2

            # Calculate accuracy for the current episode
            predictions = torch.argmax(class_probabilities, dim=1)
            accuracy = (predictions == eval_query_labels_remap).float().mean().item()
            eval_total_loss += loss.item()
            eval_total_accuracy += accuracy

            all_eval_true_labels.extend(eval_query_labels_remap.cpu().numpy())
            all_eval_predictions.extend(predictions.cpu().numpy())
            step_num += 1

    # Calculate evaluation metrics
    eval_avg_loss = eval_total_loss / conf.num_eval_episodes
    eval_avg_accuracy = eval_total_accuracy / conf.num_eval_episodes
    f1 = f1_score(all_eval_true_labels, all_eval_predictions, average='weighted')
    f_beta = fbeta_score(all_eval_true_labels, all_eval_predictions, beta=0.9, average='weighted')

    print(f"Evaluation Average Loss: {eval_avg_loss:.4f}, Average Accuracy: {eval_avg_accuracy:.4f}, F1 Score: {f1:.4f}, , F-beta: {f_beta:.4f} \n")

    if f1 > best_f1_score:
        print(f"Previous best F1 score: {best_f1_score:.4f} - current F1 score: {f1:.4f}")
    if f_beta > best_f_beta_score:
        print(f"Previous best F-beta score: {best_f_beta_score:.4f} - current F_beta score: {f_beta:.4f}")
    #     best_f1_score = f1
    #     # best_model_weights = copy.deepcopy(embedding_model.state_dict())
    #     # print("Saving new best checkpoint... \n")
    #     # torch.save(best_model_weights, os.path.join(conf.checkpoint_dir, 'best_checkpoint.pth'))

    # print(f"Evaluation complete. Best F1 score achieved is {best_f1_score:.2f}")
        
    return -f_beta

if __name__ =='__main__':

    # search_space = [
    #     Real(0.5e-6, 8e-6, prior="log-uniform", name="learning_rate"),
    #     Real(1e-5, 1e-4, prior="log-uniform", name="weight_decay"),
    #     Integer(45, 60, name="n_features"),  # Assuming you have a maximum of 60 features
    # ]

    # @use_named_args(search_space)
    # def objective(**params):
    #     # Update the conf object with the hyperparameters from the search space
    #     conf.learning_rate = params["learning_rate"]
    #     conf.weight_decay = params["weight_decay"]
    #     conf.n_features = params["n_features"]

    #     # Call your existing train_pt_classifier function
    #     negative_best_f_beta_score = train_pt_classifier(conf, train_dataset, test_dataset)

    #     # Return the negative best F1 score as we want to maximize it
    #     return negative_best_f_beta_score


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


    # Create a directory to save the best checkpoint
    os.makedirs(conf.checkpoint_dir, exist_ok=True)

    train_pt_classifier(conf, train_dataset, test_dataset)

    # result = gp_minimize(
    #     func=objective,
    #     dimensions=search_space,
    #     n_calls=10,  # Number of iterations for the optimization
    #     random_state=42,  # Ensure reproducibility
    #     n_jobs=-1,  # Use all available CPU cores
    #     )

    # print("Best score:", -result.fun)
    # print("Best parameters:", result.x)
