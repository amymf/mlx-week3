import torch
from torchvision import transforms
from datasets import load_dataset
import math

label_to_index = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    "<sos>": 10,
    "<eos>": 11,
    "<pad>": 12,
}

index_to_label = {v: k for k, v in label_to_index.items()}

ds = load_dataset("mnist")

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert PIL Image to Tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ]
)


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        label = self.dataset[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = MNISTDataset(ds["train"], transform=transform)
test_dataset = MNISTDataset(ds["test"], transform=transform)


class MNISTTiledDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, num_images, transform=None):
        if int(math.sqrt(num_images)) ** 2 != num_images:
            raise ValueError("num_images must be a perfect square (e.g. 4, 9, 16)")
        self.dataset = hf_dataset
        self.transform = transform
        self.num_images = num_images
        self.grid_size = int(math.sqrt(num_images))

    def __len__(self):
        return len(self.dataset) // self.num_images

    def __getitem__(self, idx):
        images = []
        labels = []
        grid_rows = []
        start_idx = idx * self.num_images

        for i in range(self.num_images):
            sample = self.dataset[start_idx + i]
            image = sample["image"]
            label = sample["label"]

            if self.transform:
                image = self.transform(image)

            images.append(image)
            labels.append(label_to_index[label])

        # Create each row in the grid
        for i in range(self.grid_size):
            row = torch.cat(
                images[i * self.grid_size : (i + 1) * self.grid_size], dim=2
            )  # concat width-wise
            grid_rows.append(row)

        # Stack all rows height-wise
        tiled_image = torch.cat(grid_rows, dim=1)  # concat height-wise
        input_label = torch.tensor(
            [label_to_index["<sos>"]] + labels,
            dtype=torch.long,
        )
        target = torch.tensor(
            labels + [label_to_index["<eos>"]],
            dtype=torch.long,
        )
        return tiled_image, input_label, target


train_tiled_dataset = MNISTTiledDataset(ds["train"], 4, transform=transform)
test_tiled_dataset = MNISTTiledDataset(ds["test"], 4, transform=transform)
