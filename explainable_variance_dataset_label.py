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

dataset = LoadDatasetKeypoints()
df = dataset.load_dataset_info(path_keypoints, path_label)
df_augmented = dataset.load_dataset_info(path_keypoints_augmented)
df = pd.concat([df, df_augmented], ignore_index=True)

df, label_list = map_label(df)

train_data, test_data = split_dataset(df)

### CHECK IMPACTING KEYPOINTS with VARIANCE DIFFERENCE

correct_photo = df[df["label"] == "[correct_posture]"]
correct_photo_path = list(correct_photo["path"])

incorrect_photo = df[df["label"] == "[INcorrect_posture]"]
incorrect_photo_path = list(incorrect_photo["path"])

correct = []
for file in correct_photo_path:
    file = get_keypoint_path(file)
    keypoint = load_keypoint(file, normalize_keypoint=False)
    correct.append(keypoint)
correct = torch.cat(correct)
avg_correct = correct.mean(axis=0)

incorrect = []
for file in incorrect_photo_path:
    file = get_keypoint_path(file)
    keypoint = load_keypoint(file, normalize_keypoint=False)
    incorrect.append(keypoint)
incorrect = torch.cat(incorrect)
avg_incorrect = incorrect.mean(axis=0)

drawed_img = draw_keypoints(avg_correct)
cv2.imwrite("correct_posture.jpg", drawed_img)

drawed_img = draw_keypoints(avg_incorrect)
cv2.imwrite("INcorrect_posture.jpg", drawed_img)

keypoint_variance_correct = get_keypoint_with_more_variance(correct)
keypoint_variance_incorrect = get_keypoint_with_more_variance(incorrect)

test = get_differences_between_keypoints(avg_correct, avg_incorrect)
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

var_correct = get_keypoint_with_more_variance(correct)[: len(list_impacting_keypoints)]
var_incorrect = get_keypoint_with_more_variance(incorrect)[
    : len(list_impacting_keypoints)
]

score_correct = 0
for i in range(len(list_impacting_keypoints)):
    if var_correct[i] in list_impacting_keypoints:
        score_correct += 1 * weight_keypoint[i]

score_incorrect = 0
for i in range(len(list_impacting_keypoints)):
    if var_incorrect[i] in list_impacting_keypoints:
        score_incorrect += 1 * weight_keypoint[i]

print("Score correct: ", score_correct)
print("Score incorrect: ", score_incorrect)

keypoint_variance_correct = get_keypoint_variance(correct)
keypoint_variance_incorrect = get_keypoint_variance(incorrect)

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
    avg_incorrect, avg_correct
)
print(f"Order of impacting keypoints based on direction: {joint_name_direction_diff}")
