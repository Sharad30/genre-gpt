import torch
import argparse
import json
import os

from model import GenreGPT
from utils import CharTokenizer, load_model

def generate_text(config, prompt_text, sampling_method):
    """Generates text using a trained GenreGPT model."""
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    print(f"Using device: {device} for inference.")

    # Load tokenizer
    if not os.path.exists(config.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer vocabulary not found at {config.tokenizer_path}")
    tokenizer = CharTokenizer.load_vocab(config.tokenizer_path)
    print(f"Tokenizer loaded from {config.tokenizer_path}. Vocab size: {tokenizer.vocab_size}")

    # Load model configuration used during training
    # The model architecture must match the saved checkpoint.
    model_config_path = os.path.join(os.path.dirname(config.model_path), 'config.json')
    if not os.path.exists(model_config_path):
        # Fallback to command line args if config.json is not found (e.g. using a model trained with different defaults)
        print(f"Warning: Model config.json not found at {model_config_path}. Using CLI args for model params.")
        model_params = {
            'd_model': config.d_model,
            'nhead': config.nhead,
            'num_decoder_layers': config.num_decoder_layers,
            'dim_feedforward': config.dim_feedforward,
            'max_seq_length': config.max_seq_length,
            'dropout': config.dropout
        }
    else:
        with open(model_config_path, 'r') as f:
            train_config = json.load(f)
        model_params = {
            'd_model': train_config['d_model'],
            'nhead': train_config['nhead'],
            'num_decoder_layers': train_config['num_decoder_layers'],
            'dim_feedforward': train_config['dim_feedforward'],
            'max_seq_length': train_config['max_seq_length'],
            'dropout': train_config['dropout']
        }
        print(f"Loaded model architecture parameters from {model_config_path}")

    # Initialize model
    model = GenreGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=model_params['d_model'],
        nhead=model_params['nhead'],
        num_decoder_layers=model_params['num_decoder_layers'],
        dim_feedforward=model_params['dim_feedforward'],
        max_seq_length=model_params['max_seq_length'],
        dropout=model_params['dropout']
    ).to(device)

    # Load trained model weights
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at {config.model_path}")
    
    # We only need to load model state_dict for inference, not optimizer or epoch
    checkpoint = torch.load(config.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Trained model weights loaded from {config.model_path}")

    # Generate text
    generated_plot = model.generate(
        tokenizer=tokenizer,
        prompt=prompt_text,
        max_length=config.max_generation_length,
        temperature=config.temperature,
        top_k=config.top_k,
        sampling_method=sampling_method
    )
    return generated_plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate movie plots using a trained GenreGPT model.')
    
    # Required arguments
    parser.add_argument('--prompt', type=str, required=True, help='Genre prompt (e.g., "Genre: horror\n")')
    parser.add_argument('--model_path', type=str, default='./checkpoints/genre_gpt_best_model.pth', help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer_vocab.json', help='Path to the tokenizer vocabulary file')

    # Generation parameters
    parser.add_argument('--max_generation_length', type=int, default=200, help='Maximum length of the generated plot')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling (higher means more random)')
    parser.add_argument('--top_k', type=int, default=10, help='Top-k tokens to consider for sampling')
    parser.add_argument('--sampling_method', type=str, default='top_k', choices=['top_k', 'greedy'], help='Sampling method')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use CUDA if available')

    # Model parameters (if config.json is not found with the model, these will be used)
    # These should ideally match the training configuration of the loaded model.
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension (d_model) - used if config.json not found')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads - used if config.json not found')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of decoder layers - used if config.json not found')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Feedforward dimension - used if config.json not found')
    parser.add_argument('--max_seq_length', type=int, default=200, help='Model's max sequence length during training - used if config.json not found')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate - used if config.json not found')

    args = parser.parse_args()

    print(f"Generating plot with {args.sampling_method} sampling...")
    generated_text = generate_text(args, args.prompt, args.sampling_method)
    print("\n--- Generated Plot ---")
    print(generated_text)
    print("----------------------") 