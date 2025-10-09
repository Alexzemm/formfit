import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PoseTransformer(nn.Module):
    def __init__(self, num_keypoints=33, d_model=128, num_heads=4, num_layers=2, num_classes=3):
        super().__init__()
        self.embedding = nn.Linear(num_keypoints * 3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):

        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0) 
        return self.classifier(x)



def train_model(epochs=10, batch_size=4, lr=1e-4):
    train_loader, val_loader, classes = get_dataloaders(batch_size=batch_size)
    num_classes = len(classes)

    model = PoseTransformer(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} Acc: {train_acc:.2f}%")

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"Validation Acc: {val_acc:.2f}%\n")

    torch.save(model.state_dict(), "pushup_transformer.pth")
    print("Model saved as pushup_transformer.pth")


if __name__ == "__main__":
    train_model()
