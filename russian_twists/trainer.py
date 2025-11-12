import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloaders
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# -----------------------------
# Improved Model: Pose Transformer
# -----------------------------
class PoseTransformer(nn.Module):
    def __init__(self, num_keypoints=33, d_model=256, num_heads=8, num_layers=4, num_classes=2, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        
        # Input embedding with batch normalization
        self.embedding = nn.Sequential(
            nn.Linear(num_keypoints * 3, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder with more capacity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding (need to transpose for pos_encoder)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)  # Add positional encoding
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model) back for batch_first
        
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Use both mean and max pooling over time
        x_mean = x.mean(dim=1)  # (batch, d_model)
        x_max = x.max(dim=1)[0]  # (batch, d_model)
        x = x_mean + x_max  # Combine features
        
        return self.classifier(x)


# -----------------------------
# Training Function with improvements
# -----------------------------
def train_model(epochs=50, batch_size=8, lr=5e-4, patience=10, min_delta=0.01):
    train_loader, val_loader, classes = get_dataloaders(batch_size=batch_size, augment=True)
    num_classes = len(classes)
    
    print(f"Training on device: {device}")
    print(f"Classes: {classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")

    model = PoseTransformer(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Early stopping variables
    best_val_acc = 0
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation phase
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
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_loss:.4f} Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            print(f"[*] New best validation accuracy: {best_val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break
        
        print()

    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    torch.save(model.state_dict(), "russian_transformer.pth")
    print(f"\n{'='*60}")
    print(f"Model saved as russian_transformer.pth")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_model()
