import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import get_keypoint_path, load_keypoint

class PostureDataset(Dataset):

    def __init__(self, df_set='', root_dir='', root_dir_augmented=None, transform=None, input='image'):

        self.df_set = df_set
        self.root_dir = root_dir
        self.transform = transform
        assert input == 'image' or input == 'keypoint'
        self.input = input
        self.root_dir_augmented = root_dir_augmented

    def __len__(self):
        return len(self.df_set)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.input == 'keypoint':
            keypoint_path = self.df_set["path"].iloc[idx]
            keypoint_path = get_keypoint_path(keypoint_path)
            data = load_keypoint(keypoint_path)

        # if self.input == 'image':
        #     img_path = self.df_set["frame_path"].iloc[idx]
        #     img_path = os.path.join(self.root_dir, img_path)
        #     data = load_image(img_path)

        label = self.df_set["label_id"].iloc[idx]
        label = torch.tensor(label).to(torch.float32)
        # list_image = list_image.swapaxes(0, -1)
        # list_image = list_image.view(list_image.size(0),list_image.size(1), -1, list_image.size(2), list_image.size(3))
        sample = {"data": data, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample["data"], sample["label"]


def instantiate_loaders(
    train_data,
    val_data,
    test_data,
    df_frame,
    path_original_data,
    path_augmented_data,
    batch_size=5,
    input_type="image",
):

    trainset = PostureDataset(train_data, root_dir=path_original_data, root_dir_augmented=path_augmented_data, input=input_type)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    if val_data is not None:
        valset = PostureDataset(val_data, root_dir=path_original_data, root_dir_augmented=path_augmented_data, input=input_type)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=True, num_workers=2
        )
    else:
        valloader = None

    testset = PostureDataset(test_data, root_dir=path_original_data, root_dir_augmented=path_augmented_data, input=input_type)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    label_ass = []
    for batch_idx, (data, label) in tqdm(enumerate(trainloader), total=len(trainloader)):
        label_ass.append(label.squeeze())

    if batch_size != 1:
        torch.unique(torch.cat(label_ass), return_counts=True)
    else:
        label_ass[0]    


    return trainloader, valloader, testloader
