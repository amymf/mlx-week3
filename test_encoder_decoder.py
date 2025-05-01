import torch
from torch.utils.data import DataLoader
import wandb
from dataset import test_tiled_dataset as test_dataset, label_to_index, index_to_label
from model import EncoderDecoderTransformerMNIST
import matplotlib.pyplot as plt

wandb.init(project="mnist-transformer-encoder-decoder")

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EncoderDecoderTransformerMNIST(num_classes=len(label_to_index)).to(device)

model.load_state_dict(torch.load("transformer_mnist_encoder_decoder.pth"))
model.eval()

correct = 0
total = 0
all_correct = 0

# visualisation
num_visualizations = 20
exclude_tokens = {
    label_to_index["<pad>"],
    label_to_index["<sos>"],
    label_to_index["<eos>"]
}
fig, axs = plt.subplots(1, num_visualizations, figsize=(15, 3))

with torch.no_grad():
    for batch_idx, batch in enumerate(test_dataloader):
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

        if batch_idx == 0:
            for i in range(num_visualizations):  # show first N examples
                img = images[i].cpu().squeeze().numpy()  # shape: (H, W)

                # Get the real and predicted labels, filtering out pads
                true_labels = [index_to_label[l.item()] for l, m in zip(target[i], mask[i]) if m]
                pred_labels = [index_to_label[p.item()] for p, m in zip(predicted[i], mask[i]) if m]
                true_labels = [str(index_to_label[int(idx)]) for idx in target[i] if int(idx) not in exclude_tokens]
                pred_labels = [str(index_to_label[int(idx)]) for idx in predicted[i] if int(idx) not in exclude_tokens]

                axs[i].imshow(img, cmap="gray")
                axs[i].axis("off")
                axs[i].set_title(f"T: {''.join(true_labels)}\nP: {''.join(pred_labels)}")

    plt.tight_layout()
    plt.show()

accuracy = 100 * correct / total
accuracy_all = 100 * all_correct / len(test_dataset)
wandb.log({"test_accuracy": accuracy})
print(f"Test Accuracy: {accuracy:.2f}%, All Correct: {accuracy_all:.2f}%")
