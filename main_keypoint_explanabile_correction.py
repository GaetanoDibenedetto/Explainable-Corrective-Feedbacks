import os
import numpy as np
import cv2
import pandas as pd
import torch
import random
from training.load_dataset import *
from training.dataset import *
from training.model_pipeline import *
from training.model_keypoint import *
from utils import (
    draw_keypoints,
    get_keypoint_path,
    normalize,
    get_skelton_info,
    get_joint_info,
    path_keypoints,
    path_keypoints_augmented,
    path_label,
    number_of_keypoints,
)
import random
from scipy.spatial import distance


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_model(model_path, device):
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model


def predict_single_image(model, keypoint, normalize_keypoint=True):
    if len(keypoint.shape) == 2:
        keypoint = keypoint.unsqueeze(0)

    if len(keypoint.shape) == 3:
        keypoint = keypoint.unsqueeze(0)

    keypoint = keypoint.to(device)
    keypoint = normalize(keypoint)
    output = model(keypoint)
    return torch.max(output, axis=1)[1]


def get_probability_single_image(model, keypoint, normalize_keypoint=True):
    if len(keypoint.shape) == 2:
        keypoint = keypoint.unsqueeze(0)

    if len(keypoint.shape) == 3:
        keypoint = keypoint.unsqueeze(0)

    keypoint = keypoint.to(device)
    keypoint = normalize(keypoint)
    output = model(keypoint)
    return torch.max(output, axis=1)[0].item()


def correcting_posture_random(model, keypoint):
    keypoint_original = keypoint.clone()
    count = 0
    while predict_single_image(model, keypoint) == 0:
        if count > 1000:
            return None, None
        count += 1
        keypoint = keypoint_original.clone()
        rand_keypoint = random.randint(0, keypoint.shape[0] - 1)

        x, y = get_new_random_keypoint(keypoint)

        keypoint[rand_keypoint] = torch.tensor([y, x])

    drawed_img = draw_keypoints(keypoint_original)
    cv2.imwrite("test_wrong.jpg", drawed_img)
    drawed_img = draw_keypoints(keypoint)
    cv2.imwrite("test.jpg", drawed_img)

    return keypoint, rand_keypoint


def correcting_posture_with_prefixed_keypoint(model, keypoint, keypoint_order):
    keypoint_original = keypoint.clone()

    for k in keypoint_order:
        keypoint = keypoint_original.clone()
        # keypoint = correct_posture_with_specified_keypoint(model, keypoint, k)
        keypoint = correcting_posture_percentage_increasing(model, keypoint, k)
        if keypoint is not None:
            drawed_img = draw_keypoints(keypoint_original)
            cv2.imwrite("test_wrong.jpg", drawed_img)
            drawed_img = draw_keypoints(keypoint)
            cv2.imwrite("test.jpg", drawed_img)
            print(f"KeyPoint {k} corrected")

    return keypoint


def correct_posture_with_specified_keypoint(model, keypoint, k):
    keypoint_original = keypoint.clone()
    count = 0

    while predict_single_image(model, keypoint) == 0:
        if count > 1000:
            print(f"No solution found for keypoint {k}")
            return None
        count += 1
        keypoint = keypoint_original.clone()
        # rand_keypoint = random.randint(0, keypoint.shape[0] - 1)
        rand_keypoint = k

        x, y = get_new_random_keypoint(keypoint)

        keypoint[rand_keypoint] = torch.tensor([y, x])
        return keypoint


def correcting_posture_percentage_increasing(model, keypoint, k):
    keypoint_original = keypoint.clone()
    count = 0
    while predict_single_image(model, keypoint) == 0:
        if count > 1000:
            return None
        count += 1
        keypoint = keypoint_original.clone()

        rand_keypoint = k

        # x, y = get_new_keypoint_percentage(keypoint, rand_keypoint, count)
        x, y = get_new_random_keypoint(keypoint)

        keypoint[rand_keypoint] = torch.tensor([y, x])

    drawed_img = draw_keypoints(keypoint_original)
    cv2.imwrite("test_wrong.jpg", drawed_img)
    drawed_img = draw_keypoints(keypoint)
    cv2.imwrite("test.jpg", drawed_img)

    return keypoint


def get_new_keypoint_percentage(keypoint, keypoint_idx, increment):
    # y_min, x_min = torch.abs(keypoint[keypoint_idx]) * ((100 - percentage) / 100)
    # y_max, x_max = torch.abs(keypoint[keypoint_idx]) * ((100 + percentage) / 100)
    y_min, x_min = keypoint[keypoint_idx] - increment
    y_max, x_max = keypoint[keypoint_idx] + increment
    y_min, x_min, y_max, x_max = int(y_min), int(x_min), int(y_max), int(x_max)
    y = random.randint(y_min, y_max)
    x = random.randint(x_min, x_max)
    return x, y


def get_new_random_keypoint(keypoint):
    y_min, x_min = torch.min(keypoint, axis=0)[0]
    y_max, x_max = torch.max(keypoint, axis=0)[0]
    y_min, x_min, y_max, x_max = int(y_min), int(x_min), int(y_max), int(x_max)
    y = random.randint(y_min, y_max)
    x = random.randint(x_min, x_max)
    return x, y


def get_keypoint_with_more_variance(data):
    return torch.argsort(torch.var(data.squeeze(), axis=[0, 2]), 0, descending=True)


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


def correcting_posture_respect_gt(
    model, keypoint, keypoint_gt, impacting_keypoints=None
):

    new_keypoint, keypoint_moved = get_new_keypoint_based_on_gt(
        model, keypoint, keypoint_gt, impacting_keypoints
    )
    return new_keypoint, keypoint_moved


def move_keypoint(keypoint, keypoint_gt, keypoint_idx):
    x, y = keypoint[keypoint_idx]
    x_gt, y_gt = keypoint_gt[keypoint_idx]

    x_min = torch.min(torch.tensor([x, x_gt]))
    x_max = torch.max(torch.tensor([x, x_gt]))
    y_min = torch.min(torch.tensor([y, y_gt]))
    y_max = torch.max(torch.tensor([y, y_gt]))

    y_min, x_min, y_max, x_max = int(y_min), int(x_min), int(y_max), int(x_max)
    y = random.randint(y_min, y_max)
    x = random.randint(x_min, x_max)
    keypoint[keypoint_idx] = torch.tensor([x, y])

    return keypoint


def get_new_keypoint_based_on_gt(
    model, keypoint, keypoint_gt, list_keypoint_to_move=None
):
    if list_keypoint_to_move is None:
        list_keypoint_to_move = np.arange(0, number_of_keypoints, 1).tolist()
    if predict_single_image(model, keypoint) == 1:
        print("the posture is already correct")
        return keypoint, None
    keypoint_original = keypoint.clone()
    keypoint_moved = []
    keypoint_result = []
    for keypoint_to_move in list_keypoint_to_move:
        count = 0
        test_prediction = 0
        while test_prediction == 0:
            keypoint = keypoint_original.clone()
            count += 1
            if count > 100:
                test_prediction = 1
                continue

            if keypoint_to_move in keypoint_moved:
                test_prediction = 1
                continue

            list_keypoint_to_move = keypoint_to_move
            keypoint = move_keypoint(keypoint, keypoint_gt, list_keypoint_to_move)
            test_prediction = predict_single_image(model, keypoint)

        if predict_single_image(model, keypoint) == 1:
            keypoint_result.append(keypoint)
            keypoint_moved.append(keypoint_to_move)

    if len(keypoint_result) == 0:
        return None, None
    result_idx = 0
    max = 0

    keypoint_result, keypoint_moved = augment_results(
        keypoint_result, keypoint_moved, keypoint_original
    )

    for idx, keypoint in enumerate(keypoint_result):
        drawed_img = draw_keypoints(keypoint)
        cv2.imwrite("test.jpg", drawed_img)
        drawed_img = draw_keypoints(keypoint_original)
        cv2.imwrite("test_wrong.jpg", drawed_img)

        probabilty_correct = get_probability_single_image(model, keypoint)
        if probabilty_correct > max:
            max = probabilty_correct
            result_idx = idx

    return keypoint_result[result_idx], keypoint_moved[result_idx]


def augment_results(keypoint_result, keypoint_moved, keypoint_original):
    joints = get_joint_info("coco_25")
    keypoint_result_augmented = []
    keypoint_moved_augmented = []

    for i, joint_idx in enumerate(keypoint_moved):
        joint_name = joints[joint_idx]
        if "right" in joint_name:
            opposite_joint_name = joint_name.replace("right", "left")
        elif "left" in joint_name:
            opposite_joint_name = joint_name.replace("left", "right")
        else:
            continue
        opposite_joint_idx = [k for k, v in joints.items() if v == opposite_joint_name][
            0
        ]
        keypoint = keypoint_result[i].clone()
        difference_keypoint_moved = keypoint[joint_idx] - keypoint_original[joint_idx]
        keypoint[opposite_joint_idx][0] += difference_keypoint_moved[0]
        keypoint[opposite_joint_idx][1] += difference_keypoint_moved[1] * -1
        keypoint_result_augmented.append(keypoint)
        keypoint_moved_augmented.append([joint_idx, opposite_joint_idx])

    keypoint_result = keypoint_result + keypoint_result_augmented
    keypoint_moved = keypoint_moved + keypoint_moved_augmented
    return keypoint_result, keypoint_moved


set_all_seeds(42)

SEPARATOR = os.sep


dataset = LoadDatasetKeypoints()
df = dataset.load_dataset_info(path_keypoints, path_label)
df_augmented = dataset.load_dataset_info(path_keypoints_augmented)
df = pd.concat([df, df_augmented], ignore_index=True)

df, label_list = map_label(df)

train_data, test_data = split_dataset(df)

train_label_balance = list((train_data["label_id"].value_counts() / len(train_data)))
test_label_balance = list((test_data["label_id"].value_counts() / len(test_data)))

global device
device = "cpu"
if torch.cuda.is_available():
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"


# load model
model_type = "keypoint"
model_path = os.path.join("archives_data_posture_correction", "model", model_type)
list_models = os.listdir(model_path)
list_models.sort()
model_name = list_models[0]
model = load_model(os.path.join(model_path, model_name), device)

# get correct and incorrect posture photos
correct_photo = df[df["label"] == "[correct_posture]"]
correct_photo_path = list(correct_photo["path"])

incorrect_photo = df[df["label"] == "[INcorrect_posture]"]
incorrect_photo_path = list(incorrect_photo["path"])

# from dataset, ground truth of correct posture
correct = []
for file in correct_photo_path:
    file = get_keypoint_path(file)
    keypoint = load_keypoint(file, normalize_keypoint=False)
    correct.append(keypoint)
correct = torch.cat(correct)
avg_correct = correct.mean(axis=0)


### CHECK IMPACTING KEYPOINTS MEASURING DIRECTION, CONSIDERING THE AVG KEYPOINT AS THE GT vs SINGLE INCORRECT IMAGE
test = (
    test_data.groupby(["subject", "clip_number", "frame_number", "label"])
    .agg({"path": list})
    .reset_index()
)
test_incorrect = test[test["label"] == "[INcorrect_posture]"]
test_incorrect = list(test_incorrect["path"])


joint_dict = get_joint_info("coco_25")

for i in range(100):
    test_k = []
    for file in test_incorrect[i]:
        file = get_keypoint_path(file)
        keypoint = load_keypoint(file, normalize_keypoint=False)
        test_k.append(keypoint)
    test_k = torch.cat(test_k)
    test_k = test_k.mean(axis=0)

    new_keypoint, keypoint_moved = correcting_posture_respect_gt(
        model, test_k, avg_correct
    )

    if keypoint_moved is not None:
        drawed_img = draw_keypoints(new_keypoint)
        cv2.imwrite("test.jpg", drawed_img)
        drawed_img = draw_keypoints(test_k)
        cv2.imwrite("test_wrong.jpg", drawed_img)
        if isinstance(keypoint_moved, int):
            print(f"KeyPoint {joint_dict[keypoint_moved]} corrected")
        else:
            for keypoint in keypoint_moved:
                print(f"KeyPoint {joint_dict[keypoint]} corrected")
    else:
        print("No correction found")
