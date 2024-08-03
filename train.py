# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset.dataset_loader import train_loader, test_loader, vocab, collate_batch  # Added collate_batch import
from models.baseline_model import BaselineModel
import wandb

# Hyperparameters
EMBED_DIM = 128
HIDDEN_DIM = 64
OUTPUT_DIM = 1
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

# Initialize Wandb
wandb.init(project="sentiment-analysis-baseline")

# Split training data into training and validation sets
train_dataset = list(train_loader.dataset)
train_size = int((1 - VALIDATION_SPLIT) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)

# Initialize model, loss function, and optimizer
model = BaselineModel(vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for labels, texts in train_loader:
            #print(f"Labels: {labels}")
            #print(f"Texts: {texts}")
            optimizer.zero_grad()
            outputs = model(texts).squeeze()
            #print(f"Model Outputs: {outputs}")
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.round(outputs)
            train_acc += (preds == labels).float().sum()
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for labels, texts in val_loader:
                outputs = model(texts).squeeze()
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                preds = torch.round(outputs)
                val_acc += (preds == labels).float().sum()
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        
        wandb.log({"Train Loss": train_loss, "Train Accuracy": train_acc, "Val Loss": val_loss, "Val Accuracy": val_acc})
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for labels, texts in test_loader:
            outputs = model(texts).squeeze()
            #print(f"Model Outputs (Test): {outputs}")
            loss = criterion(outputs, labels.float())
            epoch_loss += loss.item()
            preds = torch.round(outputs)
            epoch_acc += (preds == labels).float().sum()
    epoch_loss /= len(test_loader.dataset)
    epoch_acc /= len(test_loader.dataset)
    print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.4f}")

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
    evaluate_model(model, test_loader, criterion)
