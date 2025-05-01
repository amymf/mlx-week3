import torch
from torch.utils.data import DataLoader
import wandb
from dataset import ds, transform, MNISTTiledDataset, label_to_index, index_to_label
from model import EncoderDecoderTransformerMNIST
import matplotlib.pyplot as plt

wandb.init(project="mnist-transformer-encoder-decoder")

num_images = 4
batch_size = 32
test_dataset = MNISTTiledDataset(ds["test"], num_images, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderDecoderTransformerMNIST(num_classes=len(label_to_index)).to(device)
model.load_state_dict(torch.load("transformer_mnist_encoder_decoder.pth"))
model.eval()

start_sentence = torch.tensor(
    [label_to_index["<sos>"]] + [label_to_index["<pad>"]] * (num_images),
    dtype=torch.long,
).to(device)

correct = 0
total = 0
all_correct = 0

# visualisation
num_visualizations = 30
exclude_tokens = {
    label_to_index["<pad>"],
    label_to_index["<sos>"],
    label_to_index["<eos>"]
}
fig, axs = plt.subplots(1, num_visualizations, figsize=(15, 3))

for batch_idx, batch in enumerate(test_dataloader):
    images, _, target = batch
    images, target = (
        images.to(device),
        target.to(device),
    )

    start_sentences = start_sentence.unsqueeze(0).repeat(
        images.size(0), 1
    )  # shape: (batch_size, seq_len)

    # Generate output sequences one token at a time
    predictions = []
    for i in range(num_images):
        # forward
        outputs = model(images, start_sentences)
        _, predicted = torch.max(outputs.data, dim=2)
        next_token = predicted[:, i]
        predictions.append(next_token)
        start_sentences[:, i + 1] = next_token
    predictions = torch.stack(predictions, dim=1)  # shape: (batch_size, seq_len)

    mask = target != label_to_index["<pad>"]  # shape: (batch_size, seq_len)
    _, predicted = torch.max(outputs.data, dim=2)  # shape: (batch_size, seq_len)
    total += mask.sum().item()  # total number of labels excluding padding
    correct += ((predicted == target) & mask).sum().item()
    sequence_correct = ((predicted == target) | ~mask).all(dim=1)
    all_correct += sequence_correct.sum().item()

    # visualization
    mask = target != label_to_index["<pad>"]  # shape: (batch_size, seq_len)
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
print(f"Accuracy: {accuracy:.2f}%, Whole sequence accuracy: {accuracy_all:.2f}%")