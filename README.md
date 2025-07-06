# Transformer
Transformer Model for Host Depletion

### 🔧 Configuration (Before Training)

Before running the training script, update the following file paths in `train_transformer.py`:

- `log_file`: Path to save the training log file.
- `dataset_path`: Root directory where your dataset files are located.
- `train_fasta`: Filename of the training FASTA file.
- `train_labels`: Filename of the training labels file (CSV format).
- `val_fasta`: Filename of the validation FASTA file.
- `val_labels`: Filename of the validation labels file (CSV format).
- `model_save_path`: Path where the trained model (`.pth`) will be saved.

Example snippet from `train_transformer.py`:


log_file = "logs/train_log.txt"
dataset_path = "datasets/"
train_fasta = "train_seqs.fasta"
train_labels = "train_labels.csv"
val_fasta = "val_seqs.fasta"
val_labels = "val_labels.csv"
model_save_path = "models/transformer_model.pth"

Once the paths are correctly set, start training by running:
python train_transformer.py


To perform classification on a new FASTA file using the trained Transformer model, run:

python classification_transformer.py \
  --model_path example/model.pth \
  --fasta_file example/input.fasta \
  --output_csv example/prediction.csv \
  --batch_size 10000

💡 Tip: You can increase --batch_size if your system has more memory for faster processing.
