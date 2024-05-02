import os
import numpy as np
import cv2
import pandas as pd
import torch
from training.load_dataset import *
from training.dataset import *
from training.model_pipeline import *
from training.model_keypoint import *
from utils import (
    draw_keypoints,
    get_keypoint_path,
    get_skelton_info,
    get_joint_info,
    path_keypoints,
    path_keypoints_augmented,
    path_label,
    set_all_seeds,
)
from scipy.spatial import distance
import collections


def load_model(model_path, device):
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model


def get_keypoint_with_more_variance(data):
    return torch.argsort(torch.var(data.squeeze(), axis=[0, 2]), 0, descending=True)


def get_keypoint_variance(data):
    return torch.var(data.squeeze(), axis=[0, 2])


def get_differences_between_keypoints(keypoints1, keypoints2):
    skeleton = get_skelton_info("coco_25")
    link_0 = []
    link_1 = []
    dst = []
    for link in skeleton:
        try:
            k1 = (keypoints1[link[0]], keypoints1[link[1]])
            k2 = (keypoints2[link[0]], keypoints2[link[1]])
        except:
            continue
        distance_k1 = distance.euclidean(k1[0], k1[1])
        distance_k2 = distance.euclidean(k2[0], k2[1])

        link_0.append(link[0])
        link_1.append(link[1])
        dst.append(abs(distance_k1 - distance_k2))
    return {"link_0": link_0, "link_1": link_1, "distance": dst}


def get_direction_between_keypoints(keypoints1, keypoints2):
    skeleton = get_skelton_info("coco_25")
    link_0 = []
    link_1 = []
    dst = []
    for link in skeleton:
        try:
            k1 = (keypoints1[link[0]], keypoints1[link[1]])
            k2 = (keypoints2[link[0]], keypoints2[link[1]])
        except:
            continue
        direction_k1 = k1[0] - k1[1]
        direction_k2 = k2[0] - k2[1]

        link_0.append(link[0])
        link_1.append(link[1])
        dst.append((direction_k1 - direction_k2))
    return {"link_0": link_0, "link_1": link_1, "direction": dst}


def get_most_imacting_keypoint_based_on_direction(keypoint, keypoint_gt):
    test_e = get_direction_between_keypoints(keypoint, keypoint_gt)
    df = pd.DataFrame.from_dict(test_e)
    test_e_d = [t.numpy() for t in test_e["direction"]]
    avg = np.average(np.abs(test_e_d), axis=0)
    keypoint_to_change = []
    data = []
    for idx, keypoint_value in enumerate(test_e_d):
        if (np.abs(keypoint_value) > avg).sum() > 0:
            keypoint_to_change.append(idx)

    list_keypoint_idx = []
    list_direction = []
    for idx in keypoint_to_change:
        keypoint_idx = test_e["link_1"][idx]
        list_keypoint_idx.append(keypoint_idx)
        direction_to_move = test_e["direction"][idx]
        list_direction.append(direction_to_move)

    max_idx = np.argmax(np.abs(test_e_d), axis=0)
    limbs_impacting = df.iloc[max_idx][["link_0", "link_1"]].values

    avg_difference = np.mean(np.abs(test_e_d), axis=0)
    test_e_d = np.where(np.abs(test_e_d) > avg_difference, test_e_d, 0)

    index_max_difference = np.argsort(np.abs(test_e_d), axis=0)[::-1]

    joints_name = get_joint_info("coco_25")
    len_impacting_values = np.count_nonzero(test_e_d, axis=0)
    list_impacting_values = []
    for i in range(len_impacting_values.shape[0]):
        for j in range(len_impacting_values[i]):
            direction_index = index_max_difference[j][i]
            list_impacting_values.append(test_e["link_1"][direction_index])

    list_impacting_values = collections.Counter(list_impacting_values).most_common()
    list_impacting_values = [x[0] for x in list_impacting_values]
    most_imapcting_keypoints = [joints_name[int(i)] for i in list_impacting_values]

    return most_imapcting_keypoints


set_all_seeds(42)

SEPARATOR = os.sep

# preapare the dataset for the model

dataset = LoadDatasetKeypoints()
df = dataset.load_dataset_info(path_keypoints, path_label)
df_augmented = dataset.load_dataset_info(path_keypoints_augmented)
df = pd.concat([df, df_augmented], ignore_index=True)

df, label_list = map_label(df)

train_data, test_data = split_dataset(df)

global device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# load model
model_type = "keypoint"
model_path = os.path.join("archives_data_posture_correction", "model", model_type)
list_models = os.listdir(model_path)
list_models.sort()
model_name = list_models[0]
model = load_model(os.path.join(model_path, model_name), device)


# process all the test set
list_keypoint_path = list(test_data["path"])
list_data_normalized = []
list_data_not_normalized = []
for keypoint_path in list_keypoint_path:
    keypoint_path = get_keypoint_path(keypoint_path)

    list_data_normalized.append(load_keypoint(keypoint_path))
    list_data_not_normalized.append(
        load_keypoint(keypoint_path, normalize_keypoint=False)
    )

data_normalized = torch.cat(list_data_normalized).unsqueeze(dim=1).to(device)
data_not_normalized = torch.cat(list_data_not_normalized).unsqueeze(dim=1)

output = model(data_normalized)

output = output.cpu()
output = torch.max(output, axis=1)[1]


# get index of the value 1 in output
index_with_correct_posture = output == 1
index_with_uncorrect_posture = output == 0

keypoint_with_correct_posture = data_not_normalized[
    index_with_correct_posture
].squeeze()
keypoint_with_uncorrect_posture = data_not_normalized[
    index_with_uncorrect_posture
].squeeze()

avg_keypoint_path_with_correct_posture = keypoint_with_correct_posture.mean(axis=0)
drawed_img = draw_keypoints(avg_keypoint_path_with_correct_posture)
cv2.imwrite("correct_posture.jpg", drawed_img)
avg_keypoint_path_with_uncorrect_posture = keypoint_with_uncorrect_posture.mean(axis=0)
drawed_img = draw_keypoints(avg_keypoint_path_with_uncorrect_posture)
cv2.imwrite("incorrect_posture.jpg", drawed_img)


### CHECK IMPACTING KEYPOINTS with VARIANCE DIFFERENCE

keypoint_variance_correct = get_keypoint_with_more_variance(
    keypoint_with_correct_posture
)
keypoint_variance_incorrect = get_keypoint_with_more_variance(
    keypoint_with_uncorrect_posture
)

test = get_differences_between_keypoints(
    avg_keypoint_path_with_correct_posture, avg_keypoint_path_with_uncorrect_posture
)
df = pd.DataFrame.from_dict(test)
df = df.sort_values(by="distance", ascending=False)
df_impacting_keypoints = df[df["distance"] > np.mean(df["distance"])]

# check if the impacting keypoints are in the list of keypoints with more variance
impacting_keypoints = collections.Counter(
    df_impacting_keypoints["link_0"].tolist()
    + df_impacting_keypoints["link_1"].tolist()
).most_common()
list_impacting_keypoints = [x[0] for x in impacting_keypoints]
weight_keypoint = [x[1] for x in impacting_keypoints]

keypoint_variance_correct = keypoint_variance_correct[: len(list_impacting_keypoints)]
keypoint_variance_incorrect = keypoint_variance_incorrect[
    : len(list_impacting_keypoints)
]

score_correct = 0
for i in range(len(list_impacting_keypoints)):
    if keypoint_variance_correct[i] in list_impacting_keypoints:
        score_correct += 1 * weight_keypoint[i]

score_incorrect = 0
for i in range(len(list_impacting_keypoints)):
    if keypoint_variance_incorrect[i] in list_impacting_keypoints:
        score_incorrect += 1 * weight_keypoint[i]

print(f"Score correct: {score_correct} Score incorrect: {score_incorrect}")

keypoint_variance_correct = get_keypoint_variance(keypoint_with_correct_posture)
keypoint_variance_incorrect = get_keypoint_variance(keypoint_with_uncorrect_posture)

variance_diff = keypoint_variance_incorrect - keypoint_variance_correct
viariance_diff_mean = torch.mean(variance_diff)
variance_diff = np.where(variance_diff > viariance_diff_mean, variance_diff, 0)


joints_name = get_joint_info("coco_25")
test = np.argsort(-variance_diff)
joint_name_var_diff = [joints_name[int(i)] for i in np.argsort(-variance_diff)][
    : np.count_nonzero(variance_diff)
]

print(f"Order of impacting keypoints based on variance: {joint_name_var_diff}")

joint_name_distance_diff = [joints_name[int(i)] for i in list_impacting_keypoints]

print(f"Order of impacting keypoints based on distance: {joint_name_distance_diff}")


joint_name_direction_diff = get_most_imacting_keypoint_based_on_direction(
    avg_keypoint_path_with_uncorrect_posture, avg_keypoint_path_with_correct_posture
)
print(f"Order of impacting keypoints based on direction: {joint_name_direction_diff}")
