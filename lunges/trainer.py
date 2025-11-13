import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PoseTransformer(nn.Module):
    def __init__(self, num_keypoints=33, d_model=128, num_heads=4, num_layers=3, num_classes=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(num_keypoints * 3, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # Positional encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, 99)
        batch_size, seq_len, _ = x.shape
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = x + self.pos_encoding[:, :seq_len, :]  # Add positional encoding
        x = self.dropout(x)
        x = self.transformer(x)  # (batch, seq_len, d_model)
        x = x.mean(dim=1)  # Global average pooling over sequence
        return self.classifier(x)



def train_model(epochs=50, batch_size=8, lr=1e-3):
    train_loader, val_loader, classes = get_dataloaders(batch_size=batch_size)
    num_classes = len(classes)
    
    print(f"Training on device: {device}")
    print(f"Classes: {classes}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}\n")

    model = PoseTransformer(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 15

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_loss:.4f} Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "lunge_transformer.pth")
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)\n")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("Model saved as lunge_transformer.pth")


if __name__ == "__main__":
    train_model()
