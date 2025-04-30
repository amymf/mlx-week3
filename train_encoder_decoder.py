import torch
from torch.utils.data import DataLoader, random_split
from dataset import train_tiled_dataset as train_dataset, label_to_index
from model import EncoderDecoderTransformerMNIST
import wandb

wandb.init(project="mnist-transformer-encoder-decoder")

torch.manual_seed(42)  # For reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

model = EncoderDecoderTransformerMNIST(num_classes=len(label_to_index)).to(device)

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()

num_epochs = 10

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch in train_dataloader:
        images, input_labels, target = batch
        # images: (batch_size, num_images, 1, 28, 28)
        # input_labels: (batch_size, num_images + 1) = (batch_size, seq_len)
        # target: (batch_size, seq_len)
        images, input_labels, target = (
            images.to(device),
            input_labels.to(device),
            target.to(device),
        )

        # forward
        outputs = model(
            images, input_labels
        )  # shape: (batch_size, seq_len, num_classes)
        loss = criterion(outputs.view(-1, len(label_to_index)), target.view(-1))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, dim=2)  # shape: (batch_size, seq_len)
        total_train += target.numel()  # total number of labels (batch_size * seq_len)
        correct_train += (predicted == target).sum().item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch in val_dataloader:
            images, input_labels, target = batch
            images, input_labels, target = (
                images.to(device),
                input_labels.to(device),
                target.to(device),
            )

            # forward
            outputs = model(images, input_labels)
            loss = criterion(outputs.view(-1, len(label_to_index)), target.view(-1))

            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, dim=2)
            total_val += target.numel()
            correct_val += (predicted == target).sum().item()

    avg_train_loss = train_loss / len(train_dataloader)
    avg_val_loss = val_loss / len(val_dataloader)
    train_accuracy = 100 * correct_train / total_train
    val_accuracy = 100 * correct_val / total_val
    wandb.log(
        {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        }
    )
    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"Train Accuracy: {train_accuracy:.2f}%, "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

torch.save(model.state_dict(), "transformer_mnist_encoder_decoder.pth")
print("Model saved as transformer_mnist_encoder_decoder.pth")
