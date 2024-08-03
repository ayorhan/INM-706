# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset_loader import train_loader, test_loader, vocab
from models.baseline_model import BaselineModel
import wandb

# Hyperparameters
EMBED_DIM = 128
HIDDEN_DIM = 64
OUTPUT_DIM = 1
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# Initialize Wandb
wandb.init(project="sentiment-analysis-baseline")

# Initialize model, loss function, and optimizer
model = BaselineModel(vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        for labels, texts in train_loader:
            optimizer.zero_grad()
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = torch.round(outputs)
            epoch_acc += (preds == labels).float().sum()
        epoch_loss /= len(train_loader.dataset)
        epoch_acc /= len(train_loader.dataset)
        wandb.log({"Train Loss": epoch_loss, "Train Accuracy": epoch_acc})
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for labels, texts in test_loader:
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels.float())
            epoch_loss += loss.item()
            preds = torch.round(outputs)
            epoch_acc += (preds == labels).float().sum()
    epoch_loss /= len(test_loader.dataset)
    epoch_acc /= len(test_loader.dataset)
    print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.4f}")

if __name__ == "__main__":
    train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)
    evaluate_model(model, test_loader, criterion)
