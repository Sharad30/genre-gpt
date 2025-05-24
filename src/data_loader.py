import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from utils import CharTokenizer
import os

class MoviePlotDataset(Dataset):
    """PyTorch Dataset for movie plots."""
    def __init__(self, texts, tokenizer, max_seq_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate if longer than max_seq_length - 1 (to leave space for EOS if not already there or for target)
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        input_tokens = tokens[:-1] # Input is sequence up to the second to last token
        target_tokens = tokens[1:] # Target is sequence from the second token
        
        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)

def collate_fn(batch, pad_value):
    """Collate function to pad sequences in a batch."""
    inputs, targets = zip(*batch)
    
    # Pad inputs
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_value)
    
    # Pad targets
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_value)
    
    return inputs_padded, targets_padded

def prepare_datasets(csv_path, tokenizer, batch_size, max_seq_length, val_split=0.1, random_seed=42):
    """
    Loads data, creates tokenizer (if not provided and vocab doesn't exist),
    splits data, and creates DataLoaders.
    Args:
        csv_path (str): Path to the CSV file with 'Genre' and 'Plot' columns.
        tokenizer (CharTokenizer): An instance of CharTokenizer.
        batch_size (int): Batch size for DataLoaders.
        max_seq_length (int): Maximum sequence length for padding/truncation.
        val_split (float): Proportion of data to use for validation.
        random_seed (int): Random seed for splitting data.
    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    """
    df = pd.read_csv(csv_path)
    
    # Prepend genre to plot summary
    df['text'] = "Genre: " + df['Genre'] + "\n" + df['Plot']
    texts = df['text'].tolist()

    # Create dataset
    dataset = MoviePlotDataset(texts, tokenizer, max_seq_length)
    
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(random_seed))
    
    pad_value = tokenizer.char_to_idx.get('<pad>', 0) # Get pad token_id

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=lambda b: collate_fn(b, pad_value))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, pad_value))
    
    print(f"Prepared datasets: Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"Pad token ID: {pad_value}")
    
    return train_loader, val_loader

if __name__ == '__main__':
    # Example usage:
    dummy_csv_path = 'dummy_movie_plots.csv' # Make sure this file exists
    # Create a dummy csv if it doesn't exist for testing
    if not os.path.exists(dummy_csv_path):
        data = {
            'Genre': ['horror', 'sci-fi', 'romance', 'action', 'comedy'] * 20, # 100 samples
            'Plot': ["A scary plot.", "A futuristic plot.", "A love story.", "An action-packed plot.", "A funny plot."] * 20
        }
        pd.DataFrame(data).to_csv(dummy_csv_path, index=False)
        print(f"Created dummy dataset at {dummy_csv_path}")

    corpus_df = pd.read_csv(dummy_csv_path)
    corpus_texts = ("Genre: " + corpus_df['Genre'] + "\n" + corpus_df['Plot']).tolist()
    
    tokenizer_path = "./tokenizer_vocab.json"
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = CharTokenizer.load_vocab(tokenizer_path)
    else:
        print("Building tokenizer from corpus...")
        tokenizer = CharTokenizer(corpus_texts)
        tokenizer.save_vocab(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    BATCH_SIZE = 2
    MAX_SEQ_LENGTH = 200
    
    train_loader, val_loader = prepare_datasets(
        csv_path=dummy_csv_path, 
        tokenizer=tokenizer, 
        batch_size=BATCH_SIZE, 
        max_seq_length=MAX_SEQ_LENGTH
    )

    print(f"\nNumber of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Test a batch from train_loader
    print("\nTesting a batch from train_loader:")
    for i, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print("Inputs shape:", inputs.shape)      # (batch_size, seq_len)
        print("Targets shape:", targets.shape)    # (batch_size, seq_len)
        print("Sample Input 0:", inputs[0])
        print("Sample Target 0:", targets[0])
        print("Decoded Input 0:", tokenizer.decode(inputs[0].tolist(), remove_special_tokens=False))
        print("Decoded Target 0:", tokenizer.decode(targets[0].tolist(), remove_special_tokens=False))
        break
    
    # Test a batch from val_loader
    print("\nTesting a batch from val_loader:")
    for i, (inputs, targets) in enumerate(val_loader):
        print(f"Batch {i+1}:")
        print("Inputs shape:", inputs.shape)
        print("Targets shape:", targets.shape)
        print("Sample Input 0:", inputs[0])
        print("Sample Target 0:", targets[0])
        print("Decoded Input 0:", tokenizer.decode(inputs[0].tolist(), remove_special_tokens=False))
        print("Decoded Target 0:", tokenizer.decode(targets[0].tolist(), remove_special_tokens=False))
        break 