import os
import random
import cv2
import numpy as np
import torch
from skimage import io
import matplotlib.pyplot as plt

path_keypoints = os.path.join("archives_data_posture_correction", "keypoints")
path_keypoints_augmented = os.path.join(
    "archives_data_posture_correction", "keypoints_augmented"
)

path_frames = os.path.join("archives_data_posture_correction", "frames")
path_frames_augmented = os.path.join(
    "archives_data_posture_correction", "frames_augmented"
)

path_label = "archives_data_posture_correction/labels/result/labels_for_train.csv"

number_of_keypoints = 15

get_skelton_info = lambda dataset: joints_dict()[dataset]["skeleton"]
get_joint_info = lambda dataset: joints_dict()[dataset]["keypoints"]


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_keypoint_path(filename):
    if "augmented" in filename:
        keypoint_path = os.path.join(path_keypoints_augmented, filename)
    else:
        keypoint_path = os.path.join(path_keypoints, filename)
    return keypoint_path


def get_frame_path(filename):
    if "augmented" in filename:
        frame_path = os.path.join(path_frames_augmented, filename)
    else:
        frame_path = os.path.join(path_frames, filename)
    return frame_path


def load_keypoint(keypoint_path, normalize_keypoint=True):
    keypoint = np.load(keypoint_path, allow_pickle=True)
    keypoint = keypoint[:number_of_keypoints, :2]  # 15 keypoints and only x, y
    keypoint = np.array([keypoint])
    keypoint = torch.tensor(keypoint).to(torch.float32)
    if normalize_keypoint:
        keypoint = normalize(keypoint)
    return keypoint


def normalize(volume):
    """Normalize the volume"""
    # scale in a 0-1 range
    volume = (volume - torch.min(volume)) / max(
        (torch.max(volume) - torch.min(volume)), 1
    )
    return volume.to(torch.float32)


def draw_keypoints(keypoints, img=None, dataset="coco_25", confidence_threshold=0.5):

    assert not isinstance(keypoints, type(None))

    if isinstance(keypoints, torch.Tensor):
        pass
        # keypoints = keypoints.squeeze().cpu().numpy().flatten()
    if isinstance(keypoints, np.ndarray):
        keypoints = keypoints.squeeze()
    if isinstance(keypoints, str):
        keypoints = get_keypoint_path(keypoints)
        keypoints = np.load(keypoints, allow_pickle=True)
        keypoints = keypoints[:number_of_keypoints]

    if len(keypoints.shape) == 2:
        keypoints = np.array([keypoints])

    skeleton = get_skelton_info(dataset)  # joints_dict()[dataset]["skeleton"]
    joint_counter = len(joints_dict()[dataset]["keypoints"].keys())

    if img is None:
        img_shape_max = max_counter(keypoints)
        img = np.zeros((img_shape_max, img_shape_max, 3), dtype=np.uint8) * 255
    elif isinstance(img, str):
        img = get_frame_path(img)
        img = cv2.imread(img)

    img = np.array(img)[..., ::-1]  # RGB to BGR for cv2 modules

    for index_subject, keypoints_subject in enumerate(keypoints):
        if keypoints_subject.shape[1] == 2:
            keypoints_subject = np.hstack(
                (keypoints_subject, np.ones((number_of_keypoints, 1)))
            )

        if keypoints_subject.shape[0] < joint_counter:
            keypoints_subject = np.concatenate(
                [
                    keypoints_subject,
                    np.zeros(
                        (
                            joint_counter - keypoints_subject.shape[0],
                            keypoints_subject.shape[1],
                        )
                    ),
                ]
            )
        img = draw_points_and_skeleton(
            img.copy(),
            keypoints_subject,
            skeleton,
            person_index=index_subject,
            points_color_palette="gist_rainbow",
            skeleton_color_palette="jet",
            points_palette_samples=10,
            confidence_threshold=confidence_threshold,
        )
    return img[..., ::-1]  # Return RGB as original


def max_counter(a_list):
    max = 0
    for elem in a_list:
        if isinstance(elem, list) or isinstance(elem, np.ndarray):
            elem = max_counter(elem)
        if elem > max:
            max = elem
    return int(max + 1)


def draw_points_and_skeleton(
    image,
    points,
    skeleton,
    points_color_palette="tab20",
    points_palette_samples=16,
    skeleton_color_palette="Set2",
    skeleton_palette_samples=8,
    person_index=0,
    confidence_threshold=0.5,
):
    """
    Draws `points` and `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        points_color_palette: name of a matplotlib color palette
            Default: 'tab20'
        points_palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        skeleton_color_palette: name of a matplotlib color palette
            Default: 'Set2'
        skeleton_palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    image = draw_skeleton(
        image,
        points,
        skeleton,
        color_palette=skeleton_color_palette,
        palette_samples=skeleton_palette_samples,
        person_index=person_index,
        confidence_threshold=confidence_threshold,
    )
    image = draw_points(
        image,
        points,
        color_palette=points_color_palette,
        palette_samples=points_palette_samples,
        confidence_threshold=confidence_threshold,
    )
    return image


def draw_points(
    image, points, color_palette="tab20", palette_samples=16, confidence_threshold=0.5
):
    """
    Draws `points` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        color_palette: name of a matplotlib color palette
            Default: 'tab20'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid points

    """
    try:
        colors = (
            np.round(np.array(plt.get_cmap(color_palette).colors) * 255)
            .astype(np.uint8)[:, ::-1]
            .tolist()
        )
    except AttributeError:  # if palette has not pre-defined colors
        colors = (
            np.round(
                np.array(
                    plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))
                )
                * 255
            )
            .astype(np.uint8)[:, -2::-1]
            .tolist()
        )

    circle_size = max(
        1, min(image.shape[:2]) // 150
    )  # ToDo Shape it taking into account the size of the detection
    # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))

    for i, pt in enumerate(points):
        if pt[2] > confidence_threshold:
            image = cv2.circle(
                image,
                (int(pt[1]), int(pt[0])),
                circle_size,
                tuple(colors[i % len(colors)]),
                -1,
            )

    return image


def draw_skeleton(
    image,
    points,
    skeleton,
    color_palette="Set2",
    palette_samples=8,
    person_index=0,
    confidence_threshold=0.5,
):
    """
    Draws a `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        color_palette: name of a matplotlib color palette
            Default: 'Set2'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    try:
        colors = (
            np.round(np.array(plt.get_cmap(color_palette).colors) * 255)
            .astype(np.uint8)[:, ::-1]
            .tolist()
        )
    except AttributeError:  # if palette has not pre-defined colors
        colors = (
            np.round(
                np.array(
                    plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))
                )
                * 255
            )
            .astype(np.uint8)[:, -2::-1]
            .tolist()
        )

    for i, joint in enumerate(skeleton):
        pt1, pt2 = points[joint]
        if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
            image = cv2.line(
                image,
                (int(pt1[1]), int(pt1[0])),
                (int(pt2[1]), int(pt2[0])),
                tuple(colors[person_index % len(colors)]),
                2,
            )

    return image


def draw_points(
    image, points, color_palette="tab20", palette_samples=16, confidence_threshold=0.5
):
    """
    Draws `points` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        color_palette: name of a matplotlib color palette
            Default: 'tab20'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid points

    """
    try:
        colors = (
            np.round(np.array(plt.get_cmap(color_palette).colors) * 255)
            .astype(np.uint8)[:, ::-1]
            .tolist()
        )
    except AttributeError:  # if palette has not pre-defined colors
        colors = (
            np.round(
                np.array(
                    plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))
                )
                * 255
            )
            .astype(np.uint8)[:, -2::-1]
            .tolist()
        )

    circle_size = max(
        1, min(image.shape[:2]) // 150
    )  # ToDo Shape it taking into account the size of the detection
    # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))

    for i, pt in enumerate(points):
        if pt[2] > confidence_threshold:
            image = cv2.circle(
                image,
                (int(pt[1]), int(pt[0])),
                circle_size,
                tuple(colors[i % len(colors)]),
                -1,
            )

    return image


def draw_skeleton(
    image,
    points,
    skeleton,
    color_palette="Set2",
    palette_samples=8,
    person_index=0,
    confidence_threshold=0.5,
):
    """
    Draws a `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        color_palette: name of a matplotlib color palette
            Default: 'Set2'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    try:
        colors = (
            np.round(np.array(plt.get_cmap(color_palette).colors) * 255)
            .astype(np.uint8)[:, ::-1]
            .tolist()
        )
    except AttributeError:  # if palette has not pre-defined colors
        colors = (
            np.round(
                np.array(
                    plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))
                )
                * 255
            )
            .astype(np.uint8)[:, -2::-1]
            .tolist()
        )

    for i, joint in enumerate(skeleton):
        pt1, pt2 = points[joint]
        if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
            image = cv2.line(
                image,
                (int(pt1[1]), int(pt1[0])),
                (int(pt2[1]), int(pt2[0])),
                tuple(colors[person_index % len(colors)]),
                2,
            )

    return image


def joints_dict():
    joints = {
        "coco_25": {
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "neck",
                6: "left_shoulder",
                7: "right_shoulder",
                8: "left_elbow",
                9: "right_elbow",
                10: "left_wrist",
                11: "right_wrist",
                12: "left_hip",
                13: "right_hip",
                14: "hip",
                15: "left_knee",
                16: "right_knee",
                17: "left_ankle",
                18: "right_ankle",
                19: "left_big toe",
                20: "left_small_toe",
                21: "left_heel",
                22: "right_big_toe",
                23: "right_small_toe",
                24: "right_heel",
            },
            "skeleton": [
                [17, 15],
                [15, 12],
                [18, 16],
                [16, 13],
                [12, 14],
                [13, 14],
                [5, 14],
                [6, 5],
                [7, 5],
                [6, 8],
                [7, 9],
                [8, 10],
                [9, 11],
                [1, 2],
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                [17, 21],
                [18, 24],
                [19, 20],
                [22, 23],
                [19, 21],
                [22, 24],
                [5, 0],
            ],
        },
        "coco": {
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle",
            },
            "skeleton": [
                [15, 13],
                [13, 11],
                [16, 14],
                [14, 12],
                [11, 12],
                [5, 11],
                [6, 12],
                [5, 6],
                [5, 7],
                [6, 8],
                [7, 9],
                [8, 10],
                [1, 2],
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                [0, 5],
                [0, 6],
            ],
        },
        "mpii": {
            "keypoints": {
                0: "right_ankle",
                1: "right_knee",
                2: "right_hip",
                3: "left_hip",
                4: "left_knee",
                5: "left_ankle",
                6: "pelvis",
                7: "thorax",
                8: "upper_neck",
                9: "head top",
                10: "right_wrist",
                11: "right_elbow",
                12: "right_shoulder",
                13: "left_shoulder",
                14: "left_elbow",
                15: "left_wrist",
            },
            "skeleton": [
                [5, 4],
                [4, 3],
                [0, 1],
                [1, 2],
                [3, 2],
                [3, 6],
                [2, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [13, 7],
                [12, 7],
                [13, 14],
                [12, 11],
                [14, 15],
                [11, 10],
            ],
        },
        "ap10k": {
            "keypoints": {
                0: "L_Eye",
                1: "R_Eye",
                2: "Nose",
                3: "Neck",
                4: "Root of tail",
                5: "L_Shoulder",
                6: "L_Elbow",
                7: "L_F_Paw",
                8: "R_Shoulder",
                9: "R_Elbow",
                10: "R_F_Paw",
                11: "L_Hip",
                12: "L_Knee",
                13: "L_B_Paw",
                14: "R_Hip",
                15: "R_Knee",
                16: "R_B_Paw",
            },
            "skeleton": [
                [0, 1],
                [0, 2],
                [1, 2],
                [2, 3],
                [3, 4],
                [3, 5],
                [5, 6],
                [6, 7],
                [3, 8],
                [8, 9],
                [9, 10],
                [4, 11],
                [11, 12],
                [12, 13],
                [4, 14],
                [14, 15],
                [15, 16],
            ],
        },
        "apt36k": {
            "keypoints": {
                0: "L_Eye",
                1: "R_Eye",
                2: "Nose",
                3: "Neck",
                4: "Root of tail",
                5: "L_Shoulder",
                6: "L_Elbow",
                7: "L_F_Paw",
                8: "R_Shoulder",
                9: "R_Elbow",
                10: "R_F_Paw",
                11: "L_Hip",
                12: "L_Knee",
                13: "L_B_Paw",
                14: "R_Hip",
                15: "R_Knee",
                16: "R_B_Paw",
            },
            "skeleton": [
                [0, 1],
                [0, 2],
                [1, 2],
                [2, 3],
                [3, 4],
                [3, 5],
                [5, 6],
                [6, 7],
                [3, 8],
                [8, 9],
                [9, 10],
                [4, 11],
                [11, 12],
                [12, 13],
                [4, 14],
                [14, 15],
                [15, 16],
            ],
        },
        "aic": {
            "keypoints": {
                0: "right_shoulder",
                1: "right_elbow",
                2: "right_wrist",
                3: "left_shoulder",
                4: "left_elbow",
                5: "left_wrist",
                6: "right_hip",
                7: "right_knee",
                8: "right_ankle",
                9: "left_hip",
                10: "left_knee",
                11: "left_ankle",
                12: "head_top",
                13: "neck",
            },
            "skeleton": [
                [2, 1],
                [1, 0],
                [0, 13],
                [13, 3],
                [3, 4],
                [4, 5],
                [8, 7],
                [7, 6],
                [6, 9],
                [9, 10],
                [10, 11],
                [12, 13],
                [0, 6],
                [3, 9],
            ],
        },
        "wholebody": {
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle",
                17: "left_big_toe",
                18: "left_small_toe",
                19: "left_heel",
                20: "right_big_toe",
                21: "right_small_toe",
                22: "right_heel",
                23: "face-0",
                24: "face-1",
                25: "face-2",
                26: "face-3",
                27: "face-4",
                28: "face-5",
                29: "face-6",
                30: "face-7",
                31: "face-8",
                32: "face-9",
                33: "face-10",
                34: "face-11",
                35: "face-12",
                36: "face-13",
                37: "face-14",
                38: "face-15",
                39: "face-16",
                40: "face-17",
                41: "face-18",
                42: "face-19",
                43: "face-20",
                44: "face-21",
                45: "face-22",
                46: "face-23",
                47: "face-24",
                48: "face-25",
                49: "face-26",
                50: "face-27",
                51: "face-28",
                52: "face-29",
                53: "face-30",
                54: "face-31",
                55: "face-32",
                56: "face-33",
                57: "face-34",
                58: "face-35",
                59: "face-36",
                60: "face-37",
                61: "face-38",
                62: "face-39",
                63: "face-40",
                64: "face-41",
                65: "face-42",
                66: "face-43",
                67: "face-44",
                68: "face-45",
                69: "face-46",
                70: "face-47",
                71: "face-48",
                72: "face-49",
                73: "face-50",
                74: "face-51",
                75: "face-52",
                76: "face-53",
                77: "face-54",
                78: "face-55",
                79: "face-56",
                80: "face-57",
                81: "face-58",
                82: "face-59",
                83: "face-60",
                84: "face-61",
                85: "face-62",
                86: "face-63",
                87: "face-64",
                88: "face-65",
                89: "face-66",
                90: "face-67",
                91: "left_hand_root",
                92: "left_thumb1",
                93: "left_thumb2",
                94: "left_thumb3",
                95: "left_thumb4",
                96: "left_forefinger1",
                97: "left_forefinger2",
                98: "left_forefinger3",
                99: "left_forefinger4",
                100: "left_middle_finger1",
                101: "left_middle_finger2",
                102: "left_middle_finger3",
                103: "left_middle_finger4",
                104: "left_ring_finger1",
                105: "left_ring_finger2",
                106: "left_ring_finger3",
                107: "left_ring_finger4",
                108: "left_pinky_finger1",
                109: "left_pinky_finger2",
                110: "left_pinky_finger3",
                111: "left_pinky_finger4",
                112: "right_hand_root",
                113: "right_thumb1",
                114: "right_thumb2",
                115: "right_thumb3",
                116: "right_thumb4",
                117: "right_forefinger1",
                118: "right_forefinger2",
                119: "right_forefinger3",
                120: "right_forefinger4",
                121: "right_middle_finger1",
                122: "right_middle_finger2",
                123: "right_middle_finger3",
                124: "right_middle_finger4",
                125: "right_ring_finger1",
                126: "right_ring_finger2",
                127: "right_ring_finger3",
                128: "right_ring_finger4",
                129: "right_pinky_finger1",
                130: "right_pinky_finger2",
                131: "right_pinky_finger3",
                132: "right_pinky_finger4",
            },
            "skeleton": [
                [15, 13],
                [13, 11],
                [16, 14],
                [14, 12],
                [11, 12],
                [5, 11],
                [6, 12],
                [5, 6],
                [5, 7],
                [6, 8],
                [7, 9],
                [8, 10],
                [1, 2],
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6],
                [15, 17],
                [15, 18],
                [15, 19],
                [16, 20],
                [16, 21],
                [16, 22],
                [91, 92],
                [92, 93],
                [93, 94],
                [94, 95],
                [91, 96],
                [96, 97],
                [97, 98],
                [98, 99],
                [91, 100],
                [100, 101],
                [101, 102],
                [102, 103],
                [91, 104],
                [104, 105],
                [105, 106],
                [106, 107],
                [91, 108],
                [108, 109],
                [109, 110],
                [110, 111],
                [112, 113],
                [113, 114],
                [114, 115],
                [115, 116],
                [112, 117],
                [117, 118],
                [118, 119],
                [119, 120],
                [112, 121],
                [121, 122],
                [122, 123],
                [123, 124],
                [112, 125],
                [125, 126],
                [126, 127],
                [127, 128],
                [112, 129],
                [129, 130],
                [130, 131],
                [131, 132],
            ],
        },
    }
    return joints
