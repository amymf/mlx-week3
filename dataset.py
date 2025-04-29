import torch
from torchvision import transforms
from datasets import load_dataset

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
