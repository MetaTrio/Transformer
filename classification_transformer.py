import torch
import torch.nn as nn
from Bio import SeqIO
import argparse
import os

# Configuration
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

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
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

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.projection = nn.Linear(64, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=MAX_LEN // 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        x = input_ids.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# Argument parser
parser = argparse.ArgumentParser(description="Classify FASTA sequences using CNN+Transformer model.")
parser.add_argument("--model_path", required=True, help="Path to the trained model .pth file.")
parser.add_argument("--fasta_file", required=True, help="Path to input FASTA file.")
parser.add_argument("--output_csv", required=True, help="Path to output CSV file.")
parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for inference.")
args = parser.parse_args()

# Load model
model = TransformerClassifier().to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# Read and encode all sequences
sequences, ids = [], []
for record in SeqIO.parse(args.fasta_file, "fasta"):
    onehot = encode_sequence_onehot(str(record.seq), MAX_LEN)
    sequences.append(onehot)
    ids.append(record.id)

# Run inference and save results incrementally
batch_size = args.batch_size
total_batches = (len(sequences) - 1) // batch_size + 1

with open(args.output_csv, "w") as f:
    f.write("id,Predicted_Label\n")  # CSV header

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        inputs = torch.stack(batch_seqs).to(device)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist()

        # Write predictions of the current batch to CSV
        for seq_id, pred in zip(batch_ids, preds):
            f.write(f"{seq_id},{pred}\n")

        print(f"Processed batch {i // batch_size + 1} / {total_batches}")

print(f"\n✅ Predictions saved to {args.output_csv}")
