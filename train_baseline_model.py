# train_baseline_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset.dataset_loader import train_loader, test_loader, vocab, collate_batch
from models.baseline_model import BaselineModel
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Initialize Wandb
wandb.init(project="sentiment-analysis-baseline")

# Hyperparameters
EMBED_DIM = 32
HIDDEN_DIM = 64
OUTPUT_DIM = 1
NUM_EPOCHS = 5
LEARNING_RATE = 0.01
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

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
        train_preds = []
        train_labels = []
        for labels, texts in train_loader:
            optimizer.zero_grad()
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(torch.round(outputs).tolist())
            train_labels.extend(labels.tolist())
        train_loss /= len(train_loader.dataset)
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_precision = precision_score(train_labels, train_preds)
        train_recall = recall_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        train_auc = roc_auc_score(train_labels, train_preds)
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for labels, texts in val_loader:
                outputs = model(texts).squeeze()
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                val_preds.extend(torch.round(outputs).tolist())
                val_labels.extend(labels.tolist())
        val_loss /= len(val_loader.dataset)
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        wandb.log({
            "Train Loss": train_loss, 
            "Train Accuracy": train_accuracy, 
            "Train Precision": train_precision,
            "Train Recall": train_recall,
            "Train F1": train_f1,
            "Train AUC": train_auc,
            "Val Loss": val_loss, 
            "Val Accuracy": val_accuracy,
            "Val Precision": val_precision,
            "Val Recall": val_recall,
            "Val F1": val_f1,
            "Val AUC": val_auc
        })
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the model
    torch.save(model.state_dict(), "baseline_model.pth")
    wandb.save("baseline_model.pth")

# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    epoch_loss = 0
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for labels, texts in test_loader:
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels.float())
            epoch_loss += loss.item()
            test_preds.extend(torch.round(outputs).tolist())
            test_labels.extend(labels.tolist())
    epoch_loss /= len(test_loader.dataset)
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_preds)
    print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    wandb.log({
        "Test Loss": epoch_loss, 
        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1": test_f1,
        "Test AUC": test_auc
    })

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
    evaluate_model(model, test_loader, criterion)
