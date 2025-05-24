import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import pandas as pd
import json

from model import GenreGPT
from data_loader import prepare_datasets
from utils import CharTokenizer, save_model, load_model

def train(config):
    """Trains the GenreGPT model."""
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    print(f"Using device: {device}")

    # Prepare tokenizer
    tokenizer_path = config.tokenizer_path
    corpus_texts = []
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = CharTokenizer.load_vocab(tokenizer_path)
    else:
        print("Building tokenizer from corpus...")
        if not os.path.exists(config.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {config.dataset_path} for building tokenizer.")
        corpus_df = pd.read_csv(config.dataset_path)
        corpus_texts = ("Genre: " + corpus_df['Genre'] + "\n" + corpus_df['Plot']).tolist()
        tokenizer = CharTokenizer(corpus_texts)
        tokenizer.save_vocab(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    pad_token_id = tokenizer.char_to_idx.get('<pad>', 0)
    print(f"Padding token ID will be: {pad_token_id}")

    # Prepare data
    train_loader, val_loader = prepare_datasets(
        csv_path=config.dataset_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_seq_length=config.max_seq_length
    )

    # Initialize model
    model = GenreGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id) # Ignore padding for loss calculation

    start_epoch = 0
    best_val_loss = float('inf')

    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        print(f"Loading checkpoint from {config.checkpoint_path}")
        start_epoch, best_val_loss = load_model(config.checkpoint_path, model, optimizer)
        start_epoch += 1 # Start from the next epoch
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")

    print(f"Model summary:")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Max sequence length: {config.max_seq_length}")
    print(f"  Embedding dimension (d_model): {config.d_model}")
    print(f"  Number of heads (nhead): {config.nhead}")
    print(f"  Number of decoder layers: {config.num_decoder_layers}")
    print(f"  Feedforward dimension: {config.dim_feedforward}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params/1e6:.2f}M")


    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            
            # Create padding mask for inputs
            # True for padded tokens, False for actual tokens
            src_key_padding_mask = (inputs == pad_token_id).to(device)

            outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)
            
            # Reshape for CrossEntropyLoss: 
            # Outputs: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
            # Targets: (batch_size, seq_len) -> (batch_size * seq_len)
            loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm) # Gradient clipping
            optimizer.step()
            total_train_loss += loss.item()

            if (batch_idx + 1) % config.log_interval == 0:
                print(f'Epoch [{epoch+1}/{config.num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{config.num_epochs}] Training Loss: {avg_train_loss:.4f}')

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                src_key_padding_mask = (inputs == pad_token_id).to(device)
                outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)
                loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.view(-1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{config.num_epochs}] Validation Loss: {avg_val_loss:.4f}')

        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, optimizer, epoch, avg_val_loss, config.best_model_path)
            print(f"Saved new best model to {config.best_model_path} (Val Loss: {best_val_loss:.4f})")
        
        # Save latest checkpoint
        save_model(model, optimizer, epoch, avg_val_loss, config.checkpoint_path)
        print(f"Saved checkpoint to {config.checkpoint_path}")

    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GenreGPT model')
    # Data and Tokenizer
    parser.add_argument('--dataset_path', type=str, default='dummy_movie_plots.csv', help='Path to the dataset CSV file')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer_vocab.json', help='Path to save/load tokenizer vocab')
    
    # Model Hyperparameters
    parser.add_argument('--d_model', type=int, default=128, help='Embedding dimension and model dimension') # Reduced for faster training
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads') # Reduced
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of decoder layers') # Reduced
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Dimension of feedforward network') # Reduced
    parser.add_argument('--max_seq_length', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training Configuration
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs') # Reduced for quick test
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm value')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use CUDA if available')

    # Checkpoints and Logging
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/genre_gpt_checkpoint.pth', help='Path to save training checkpoints')
    parser.add_argument('--best_model_path', type=str, default='./checkpoints/genre_gpt_best_model.pth', help='Path to save the best model')
    parser.add_argument('--log_interval', type=int, default=10, help='Log training progress every N batches')

    args = parser.parse_args()

    # Create checkpoint directories if they don't exist
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.best_model_path), exist_ok=True)

    # Save config
    config_path = os.path.join(os.path.dirname(args.checkpoint_path), 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved to {config_path}")

    train(args) 