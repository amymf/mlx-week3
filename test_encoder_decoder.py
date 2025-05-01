import torch
from torch.utils.data import DataLoader
import wandb
from dataset import test_tiled_dataset as test_dataset, label_to_index
from model import EncoderDecoderTransformerMNIST

wandb.init(project="mnist-transformer-encoder-decoder")

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EncoderDecoderTransformerMNIST(num_classes=len(label_to_index)).to(device)

model.load_state_dict(torch.load("transformer_mnist_encoder_decoder.pth"))
model.eval()

correct = 0
total = 0
all_correct = 0

with torch.no_grad():
    for batch in test_dataloader:
        images, input_labels, target = batch
        images, input_labels, target = (
            images.to(device),
            input_labels.to(device),
            target.to(device),
        )

        # forward
        outputs = model(images, input_labels)

        # Calculate accuracy
        mask = target != label_to_index["<pad>"]  # shape: (batch_size, seq_len)
        _, predicted = torch.max(outputs.data, dim=2)  # shape: (batch_size, seq_len)
        total += mask.sum().item()  # total number of labels excluding padding
        correct += ((predicted == target) & mask).sum().item()
        sequence_correct = ((predicted == target) | ~mask).all(dim=1)
        all_correct += sequence_correct.sum().item()

accuracy = 100 * correct / total
accuracy_all = 100 * all_correct / len(test_dataset)
wandb.log({"test_accuracy": accuracy})
print(f"Test Accuracy: {accuracy:.2f}%, All Correct: {accuracy_all:.2f}%")
