import json
import torch
import os

class CharTokenizer:
    """
    A simple character-level tokenizer.
    """
    def __init__(self, corpus=None):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        if corpus:
            self._build_vocab(corpus)

    def _build_vocab(self, corpus):
        """
        Builds the vocabulary from a given corpus.
        Args:
            corpus (list of str): A list of text strings.
        """
        chars = sorted(list(set("".join(corpus))))
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        self.vocab_size = len(chars)
        # Add special tokens if not present
        for token in ['<pad>', '<sos>', '<eos>', '<unk>']:
            if token not in self.char_to_idx:
                idx = self.vocab_size
                self.char_to_idx[token] = idx
                self.idx_to_char[idx] = token
                self.vocab_size +=1


    def encode(self, text, add_special_tokens=True):
        """
        Encodes a text string into a list of token IDs.
        Args:
            text (str): The input text string.
            add_special_tokens (bool): Whether to add <sos> and <eos> tokens.
        Returns:
            list of int: The encoded token IDs.
        """
        tokens = [self.char_to_idx.get(char, self.char_to_idx['<unk>']) for char in text]
        if add_special_tokens:
            tokens = [self.char_to_idx['<sos>']] + tokens + [self.char_to_idx['<eos>']]
        return tokens

    def decode(self, token_ids, remove_special_tokens=True):
        """
        Decodes a list of token IDs back into a text string.
        Args:
            token_ids (list of int): The list of token IDs.
            remove_special_tokens (bool): Whether to remove <sos>, <eos>, <pad> tokens.
        Returns:
            str: The decoded text string.
        """
        chars = []
        for token_id in token_ids:
            char = self.idx_to_char.get(token_id, '<unk>')
            if remove_special_tokens and char in ['<sos>', '<eos>', '<pad>', '<unk>']:
                if char == '<unk>' and self.idx_to_char.get(token_id) is None: # only skip if it was truly an unknown id
                    continue
                elif char != '<unk>': # always skip sos, eos, pad
                     continue
            chars.append(char)
        return "".join(chars)

    def save_vocab(self, filepath):
        """Saves the vocabulary to a file."""
        vocab_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {int(k):v for k,v in self.idx_to_char.items()}, # JSON keys must be strings
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=4)

    @classmethod
    def load_vocab(cls, filepath):
        """Loads the vocabulary from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        tokenizer = cls()
        tokenizer.char_to_idx = vocab_data['char_to_idx']
        # Convert keys back to int for idx_to_char
        tokenizer.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        tokenizer.vocab_size = vocab_data['vocab_size']
        return tokenizer

def save_model(model, optimizer, epoch, loss, filepath):
    """
    Saves the model, optimizer state, epoch, and loss.
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): The current epoch.
        loss (float): The current loss.
        filepath (str): Path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, model, optimizer=None):
    """
    Loads the model and optimizer state.
    Args:
        filepath (str): Path to the checkpoint.
        model (torch.nn.Module): The model instance to load state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance to load state into.
    Returns:
        tuple: epoch, loss
    """
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return 0, float('inf')
        
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {filepath}. Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss

def top_k_sampling(logits, k=5, temperature=1.0):
    """
    Samples an index from the logits using top-k sampling.
    Args:
        logits (torch.Tensor): The logits from the model (batch_size, vocab_size).
        k (int): The number of top logits to consider.
        temperature (float): Softmax temperature.
    Returns:,
        torch.Tensor: The sampled token indices (batch_size, 1).
    """
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    probabilities = torch.softmax(top_k_logits, dim=-1)
    sampled_indices_in_top_k = torch.multinomial(probabilities, 1)
    return torch.gather(top_k_indices, -1, sampled_indices_in_top_k)

def greedy_sampling(logits):
    """
    Samples an index from the logits using greedy sampling.
    Args:
        logits (torch.Tensor): The logits from the model (batch_size, vocab_size).
    Returns:
        torch.Tensor: The sampled token indices (batch_size, 1).
    """
    return torch.argmax(logits, dim=-1, keepdim=True) 