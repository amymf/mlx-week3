import torch
from torch.utils.data import DataLoader
import wandb
from dataset import test_dataset
from model import TransformerMNIST

wandb.init(project="mnist-transformer-encoder")

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerMNIST(
    image_size=28, patch_size=14, model_dim=64, num_heads=8, ff_dim=512, num_layers=4
).to(device)
model.load_state_dict(torch.load("transformer_mnist.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for batch in test_dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # forward
        outputs = model(images)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
wandb.log({"test_accuracy": accuracy})

print(f"Test Accuracy: {accuracy:.2f}%")