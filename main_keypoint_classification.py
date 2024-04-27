import os
import numpy as np
import pandas as pd
import torch
import random

from training.dataset import instantiate_loaders
from training.load_dataset import *
from training.model_pipeline import *
from training.report import print_report
from training.model_keypoint import *

from utils import path_keypoints, path_keypoints_augmented, path_label, set_all_seeds


set_all_seeds(42)

SEPARATOR = os.sep

# Load dataset
dataset = LoadDatasetKeypoints()
df = dataset.load_dataset_info(path_keypoints, path_label)
df_augmented = dataset.load_dataset_info(path_keypoints_augmented)
df_augmented = df_augmented[df_augmented["label"] == "[correct_posture]"]
df = pd.concat([df, df_augmented], ignore_index=True)

df, label_list = map_label(df)


train_data, test_data = split_dataset(df)

train_label_balance = list((train_data["label_id"].value_counts() / len(train_data)))
test_label_balance = list((test_data["label_id"].value_counts() / len(test_data)))


trainloader, valloader, testloader = instantiate_loaders(
    train_data,
    None,
    test_data,
    df,
    path_keypoints,
    path_keypoints_augmented,
    batch_size=20,
    input_type="keypoint",
)

global device
device = "cpu"
if torch.cuda.is_available():
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"

model = MLP(num_classes=len(label_list)).to(device)
# model = CNN_keypoint(num_classes=len(label_list)).to(device)

truelabels, predictions = run_pipeline(
    model,
    trainloader,
    valloader,
    testloader,
    epochs=1000,
    learning_rate=0.001,
    weight_loss=train_label_balance,
    device=device,
    model_type="keypoint",
)

print_report(truelabels, predictions)
