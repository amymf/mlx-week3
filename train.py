import torch
from torch.utils.data import DataLoader, random_split
from dataset import train_dataset
from model import TransformerMNIST
import wandb

wandb.init(project="mnist-transformer-encoder")

torch.manual_seed(42)  # For reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

model = TransformerMNIST(
    image_size=28, patch_size=14, model_dim=64, num_heads=8, ff_dim=512, num_layers=4
).to(device)

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
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch in val_dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

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
        f"Train Accuracy: {train_accuracy:.2f}%, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

torch.save(model.state_dict(), "transformer_mnist.pth")
print("Model saved as transformer_mnist.pth")
