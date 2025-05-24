import torch
import torch.nn as nn
import math
from utils import top_k_sampling, greedy_sampling

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GenreGPT(nn.Module):
    """A GPT-style decoder-only Transformer model for genre-conditioned text generation."""
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(GenreGPT, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_seq_length = max_seq_length

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass of the GenreGPT model.
        Args:
            src (torch.Tensor): Input tensor (batch_size, seq_len).
            src_mask (torch.Tensor, optional): Causal mask for the decoder.
            src_key_padding_mask (torch.Tensor, optional): Mask for padding tokens.
        Returns:
            torch.Tensor: Output logits (batch_size, seq_len, vocab_size).
        """
        src = self.embedding(src) * math.sqrt(self.d_model) # (batch_size, seq_len, d_model)
        src = src.transpose(0,1) # (seq_len, batch_size, d_model) for positional encoding
        src = self.pos_encoder(src) 
        src = src.transpose(0,1) # (batch_size, seq_len, d_model) back for transformer decoder

        if src_mask is None:
            device = src.device
            src_mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
        
        # TransformerDecoder expects target, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask
        # In a decoder-only setup, src acts as both target and memory.
        # The causal mask is tgt_mask.
        # src_key_padding_mask is tgt_key_padding_mask.
        output = self.transformer_decoder(tgt=src, memory=src, tgt_mask=src_mask, tgt_key_padding_mask=src_key_padding_mask)
        output = self.fc_out(output)
        return output

    def generate(self, tokenizer, prompt="Genre: horror\n", max_length=100, temperature=1.0, top_k=5, sampling_method='top_k'):
        """
        Generates text based on a prompt.
        Args:
            tokenizer: The tokenizer instance.
            prompt (str): The input prompt string.
            max_length (int): Maximum length of the generated sequence.
            temperature (float): Softmax temperature for sampling.
            top_k (int): Value for top-k sampling.
            sampling_method (str): 'top_k' or 'greedy'.
        Returns:
            str: The generated text.
        """
        self.eval()
        device = next(self.parameters()).device
        
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False) # No SOS/EOS for prompt here
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        generated_tokens = prompt_tokens.copy()

        with torch.no_grad():
            for _ in range(max_length - len(prompt_tokens)):
                # Ensure input_ids does not exceed max_seq_length for the model
                current_input_ids = input_ids[:, -self.max_seq_length:]
                
                # Create padding mask if needed (though not typical for generation with single sequence)
                padding_mask = (current_input_ids == tokenizer.char_to_idx.get('<pad>', -1))
                if not padding_mask.any(): # if no padding, pass None
                    padding_mask = None

                outputs = self.forward(current_input_ids, src_key_padding_mask=padding_mask)
                logits = outputs[:, -1, :]  # Get logits for the last token

                if sampling_method == 'top_k':
                    next_token_id = top_k_sampling(logits, k=top_k, temperature=temperature)
                elif sampling_method == 'greedy':
                    next_token_id = greedy_sampling(logits)
                else:
                    raise ValueError("Invalid sampling_method. Choose 'top_k' or 'greedy'.")
                
                next_token_id_item = next_token_id.item()
                generated_tokens.append(next_token_id_item)
                
                # Append the new token to input_ids for the next step
                input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0)), dim=1) # Add as new row then will be unsqueezed

                if next_token_id_item == tokenizer.char_to_idx.get('<eos>', -1):
                    break
        
        return tokenizer.decode(generated_tokens, remove_special_tokens=True) 