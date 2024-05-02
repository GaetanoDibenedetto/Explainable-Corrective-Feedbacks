# Explainable-Corrective-Feedbacks
Human Pose Estimation for Explainable Corrective Feedbacks in Office Spaces

## Compatibility
Tested on Ubuntu 23.10 with Python 3.11.6.


## Installation

1. **PyTorch**: Install PyTorch by following the instructions at [https://pytorch.org/](https://pytorch.org/).

2. **Requirements**: Install additional dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

3. **Ready to Use!**


## Reproducibility of Paper Results
The dataset can be downloaded from the following link: [CLICK HERE](https://zenodo.org/records/11075018).

The 3 folders must be inserted in a root folder of the project, after of the git clone of this repository, named `archives_data_posture_correction`, as in the tree structure following below:
```
archives_data_posture_correction
│   
├───keypoints
│       ap_1_250.jpg.npy
│		...
│		...
│       ms_3_51581.jpg.npy
│       
├───keypoints_augmented
│       augmented_ap_1_250.jpg.npy
│		...
│		...
│       augmented_ms_3_51581.jpg.npy
│       
└───labels
    └───result
            labels_for_train.csv
```            

### Execution
To reproduce the results mentioned in the associated paper, the following scripts can be utilized in the following order:

1. **Pose Classification Model (Section 4.2)**:
  - `main_keypoint_classification.py`

2. **Explaination Module based on Data Statistics (Section 4.3)**:
  - `explaination_based_on_data_statistics.py`

3. **Personalized Feedback Module (Section 4.4)**:
  - `main_keypoint_explanabile_correction.py`

## References

Special thanks to the authors of [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) for sharing their code under the Apache License 2.0. Portions of their code were reused for the keypoint drawing functionality in the `utils.py` file.

