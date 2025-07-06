## transformer with CNN layer
# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import pandas as pd
import logging
import time
import psutil
import os

# Logging setup
log_file = "/kaggle/working/logs/training_onehot_cnn_transformer.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
def log(message):
    print(message)
    logging.info(message)
    for handler in logging.getLogger().handlers:
        handler.flush()
def log_system_usage():
    max_ram = psutil.virtual_memory().used / (1024 ** 3)
    log(f"Max RAM Usage: {max_ram:.2f} GB")
def log_time(start_time):
    elapsed_time = time.time() - start_time
    log(f"Elapsed Time: {elapsed_time:.2f} seconds")

# Configuration
BATCH_SIZE = 512
EPOCHS = 50
LR = 1e-3
MAX_LEN = 512
NUM_CLASSES = 2
NUM_BASES = 4
BASE_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One-hot encoding
def encode_sequence_onehot(sequence, max_len):
    sequence = sequence.upper()
    one_hot = torch.zeros((max_len, NUM_BASES), dtype=torch.float)
    for i, base in enumerate(sequence[:max_len]):
        if base in BASE_TO_INDEX:
            one_hot[i, BASE_TO_INDEX[base]] = 1
    return one_hot

# Dataset
class SequenceDataset(Dataset):
    def __init__(self, fasta_file, label_file, max_len=MAX_LEN):
        self.max_len = max_len
        self.labels = pd.read_csv(label_file)
        self.labels = dict(zip(self.labels['ID'], self.labels['y_true']))
        self.sequences = []
        self.targets = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            if record.id in self.labels:
                encoded_seq = encode_sequence_onehot(str(record.seq), self.max_len)
                self.sequences.append(encoded_seq)
                self.targets.append(self.labels[record.id])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "input_ids": self.sequences[idx],
            "label": torch.tensor(self.targets[idx], dtype=torch.long)
        }

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):  # reduced after pooling
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

# CNN + Transformer Classifier
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=NUM_BASES, embed_dim=128, num_heads=4, ff_dim=256, num_layers=4, num_classes=NUM_CLASSES):
        super().__init__()

        # CNN
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)  # Reduces sequence length

        # Project CNN output to Transformer dimension
        self.projection = nn.Linear(64, embed_dim)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=MAX_LEN // 2)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        x = input_ids.permute(0, 2, 1)       # [B, 4, L]
        x = self.relu(self.conv1(x))         # [B, 64, L]
        x = self.pool(x)                     # [B, 64, L//2]
        x = x.permute(0, 2, 1)               # [B, L//2, 64]
        x = self.projection(x)               # [B, L//2, embed_dim]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)                    # Global average pooling
        return self.fc(x)

# Load datasets
dataset_path = "/kaggle/working/dataset"
train_fasta = os.path.join(dataset_path, "AMAISE_250000_human.fasta")
train_labels = os.path.join(dataset_path, "AMAISE_250000_true_labels_human_binary.csv")
val_fasta = os.path.join(dataset_path, "AMAISE_250000_validation_human.fasta")
val_labels = os.path.join(dataset_path, "AMAISE_250000_validation_true_labels_human_binary.csv")

train_dataset = SequenceDataset(train_fasta, train_labels)
val_dataset = SequenceDataset(val_fasta, val_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Initialize model, loss, optimizer
model = TransformerClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# Validation
def validate_model(model, val_loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch['input_ids'].to(device), batch['label'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(val_loader), correct / total

# Training loop
def train_model(model, train_loader, val_loader, epochs):
    log("Starting training...")
    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, labels = batch['input_ids'].to(device), batch['label'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        val_loss, val_acc = validate_model(model, val_loader)
        log(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, Train Acc={correct/total:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = "/kaggle/working/model/metagenomics_classifier_onehot_cnn_transformer.pth" 
            torch.save(model.state_dict(), model_save_path)
            log(f"Best model saved at {model_save_path}")

    log_time(start_time)
    log_system_usage()

# Train the model
train_model(model, train_loader, val_loader, EPOCHS)
