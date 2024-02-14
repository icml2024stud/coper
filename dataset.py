from torch.utils.data import Dataset
import torch
import numpy as np
import random
import pickle
from sklearn.utils import shuffle


class Multiview(Dataset):
    """
    Creates a multiview dataset object
    cfg.dataset_dir: A dictionary with 6 keys -
        - "dataset_name": The dataset name
        - "dataset_version": The version number of the dataset (in case it is modyfied)
        - "X": A dictionary with the raw data, each key, view_i is a numpy array with of shape (n, p_i)
        - "Y": A numpy array of labels
        - "view_names": A list of corresponding names to each view
        - "sub_sample": A list with two values, the first is a boolean indicating weather to subsample or not. The second is the amound to subsample
    """

    def __init__(self, cfg):
        super().__init__()

        # Loading the dictionary back from the pickle file
        with open(cfg.dataset_path, "rb") as file:
            dataset_dict = pickle.load(file)

        self.views = dataset_dict["X"]
        self.labels = dataset_dict["Y"]
        self.dataset_name = dataset_dict["dataset_name"]

        if dataset_dict["sub_sample"][0] == True:
            self.num_of_sub_samples = dataset_dict["sub_sample"][1]
            self.sub_sample()
        print("Number of views:", len(self.views))
        print("Dimensions:", [v.shape[1] for v in self.views.values()])
        print("Unique labels:", np.unique(list(self.labels.values())[0]))

    def __getitem__(self, index: int):
        if self.dataset_name == "noisy_mnist":
            # Given a target from view1, find a random index from view2 where the target is the same
            target = self.labels["view1"][index]
            indices = np.where(self.labels["view2"] == target)[0]
            target_idx_view2 = random.choice(indices.tolist())
            return (
                torch.tensor(self.views["view1"][index]).float(),
                torch.tensor(self.labels["view1"][index]).long(),
                torch.tensor(self.views["view2"][target_idx_view2]).float(),
                torch.tensor(self.labels["view2"][target_idx_view2]).long(),
            )
        else:
            out = []
            for i in range(len(self.views.keys())):
                if isinstance(self.views[f"view{i + 1}"][index], np.ndarray):
                    out.append(
                        torch.tensor(self.views[f"view{i + 1}"][index].reshape(-1), dtype=torch.float32))
                else:
                    out.append(
                        torch.tensor(self.views[f"view{i + 1}"][index].toarray().reshape(-1), dtype=torch.float32))
                out.append(torch.tensor(self.labels[f"view{i + 1}"][index]).long())
            return out

    def __len__(self) -> int:
        return self.views["view1"].shape[0]

    def num_classes(self):
        return np.unique(self.labels["view1"]).shape[0]

    def num_features(self):
        out = [self.views[f"view{i + 1}"].shape[1] for i in range(len(self.views.keys()))]
        return out

    def sub_sample(self):
        chosen_samples, _ = self.pick_samples()
        for view in self.views.keys():
            self.views[view] = self.views[view][chosen_samples,]

    def pick_samples(self):
        unique_labels = np.unique(self.labels)
        num_labels = len(unique_labels)

        samples_per_label = self.num_of_sub_samples // num_labels

        chosen_samples = []
        left_out_samples = []

        for label in unique_labels:
            indices = np.where(self.labels == label)[0]

            if len(indices) < samples_per_label:
                chosen_samples.extend(indices)
            else:
                np.random.shuffle(indices)
                chosen_indices = indices[:samples_per_label]
                left_out_indices = indices[samples_per_label:]
                chosen_samples.extend(chosen_indices)
                left_out_samples.extend(left_out_indices)

        chosen_samples = shuffle(chosen_samples)
        left_out_samples = shuffle(left_out_samples)

        return chosen_samples, left_out_samples
