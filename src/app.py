import streamlit as st
import torch
import argparse
import json
import os
import sys

# Add project root to Python path to allow direct imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import GenreGPT
from utils import CharTokenizer
# Assuming inference.py's generate_text can be refactored or its core logic used here
# For simplicity, we'll reimplement a focused generation function or adapt from inference.py

# --- Configuration & Model Loading ---
DEFAULT_MODEL_DIR = "./checkpoints/"
DEFAULT_MODEL_NAME = "genre_gpt_best_model.pth" # or genre_gpt_checkpoint.pth
DEFAULT_TOKENIZER_NAME = "tokenizer_vocab.json"
DEFAULT_CONFIG_NAME = "config.json"

def load_inference_dependencies(model_dir=DEFAULT_MODEL_DIR, model_name=DEFAULT_MODEL_NAME, tokenizer_name=DEFAULT_TOKENIZER_NAME, config_name=DEFAULT_CONFIG_NAME):
    """Loads tokenizer, model configuration, and the trained model."""
    tokenizer_path = os.path.join(model_dir, tokenizer_name)
    model_config_path = os.path.join(model_dir, config_name)
    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(tokenizer_path):
        st.error(f"Tokenizer vocabulary not found at {tokenizer_path}")
        return None, None, None
    tokenizer = CharTokenizer.load_vocab(tokenizer_path)

    if not os.path.exists(model_config_path):
        st.error(f"Model config.json not found at {model_config_path}. Cannot determine model architecture.")
        # As a fallback, could try to define a default architecture if CLI args were available
        # For a Streamlit app, it's better to rely on the saved config.
        return tokenizer, None, None
    
    with open(model_config_path, 'r') as f:
        train_config = json.load(f)
    
    model_params = {
        'vocab_size': tokenizer.vocab_size, # Vocab size comes from tokenizer
        'd_model': train_config.get('d_model', 128), # Provide defaults if not in config
        'nhead': train_config.get('nhead', 4),
        'num_decoder_layers': train_config.get('num_decoder_layers', 2),
        'dim_feedforward': train_config.get('dim_feedforward', 256),
        'max_seq_length': train_config.get('max_seq_length', 200),
        'dropout': train_config.get('dropout', 0.1)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GenreGPT(**model_params).to(device)

    if not os.path.exists(model_path):
        st.error(f"Trained model checkpoint not found at {model_path}")
        return tokenizer, model_params, None # Return params for info, but no model
        
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return tokenizer, model_params, None

    return tokenizer, model_params, model, device


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="GenreGPT: Movie Plot Generator")

st.title("🎬 GenreGPT: Movie Plot Generator")
st.markdown("Choose a genre and let the AI generate a movie plot for you! Compare different sampling methods.")

# Load model and dependencies
# Allow user to specify model directory (optional, advanced)
model_dir_input = st.sidebar.text_input("Model Directory (optional):", DEFAULT_MODEL_DIR)

if model_dir_input and os.path.isdir(model_dir_input):
    current_model_dir = model_dir_input
else:
    if model_dir_input and not os.path.isdir(model_dir_input): # if path provided but not valid
        st.sidebar.warning(f"Directory '{model_dir_input}' not found. Using default '{DEFAULT_MODEL_DIR}'.")
    current_model_dir = DEFAULT_MODEL_DIR


# Attempt to load model assets
try:
    tokenizer, model_cfg, model, device = load_inference_dependencies(model_dir=current_model_dir)
except TypeError: # Handles case where load_inference_dependencies returns None for some values
    st.error("Failed to load critical model components. Please check the model directory and file paths.")
    tokenizer, model_cfg, model, device = None, None, None, None


if tokenizer and model and model_cfg and device :
    st.sidebar.success("Model and Tokenizer Loaded Successfully!")
    st.sidebar.markdown("### Model Configuration:")
    st.sidebar.json(model_cfg, expanded=False)

    # Genre selection
    genres = ["horror", "sci-fi", "romance", "action", "comedy", "drama", "fantasy", "mystery", "thriller"]
    selected_genre = st.selectbox("1. Select a Movie Genre:", genres, index=0)
    
    prompt_text = f"Genre: {selected_genre.lower()}\n"
    st.text_area("Generated Prompt:", prompt_text, height=50, disabled=True)

    # Generation parameters
    st.sidebar.markdown("---")
    st.sidebar.header("Generation Parameters")
    max_gen_len = st.sidebar.slider("Max Generation Length:", min_value=50, max_value=500, value=150, step=10)
    temperature = st.sidebar.slider("Temperature (for Top-K):", min_value=0.1, max_value=2.0, value=0.8, step=0.05)
    top_k_val = st.sidebar.slider("Top-K Value:", min_value=1, max_value=50, value=10, step=1)

    if st.button("✨ Generate Plot"):
        with st.spinner("AI is conjuring a plot..."):
            st.subheader(f"Generated Plot (Top-K Sampling, T={temperature}, K={top_k_val})")
            
            # Color code for genre (simple example)
            genre_color_map = {
                "horror": "red", "sci-fi": "blue", "romance": "pink", 
                "action": "orange", "comedy": "yellow", "drama": "purple",
                "fantasy": "green", "mystery": "gray", "thriller": "black"
            }
            
            # Highlight genre in prompt
            display_prompt = f"Genre: <font color='{genre_color_map.get(selected_genre.lower(), 'black')}'><strong>{selected_genre.lower()}</strong></font>\n"

            # Generate with Top-K
            generated_text_top_k = model.generate(
                tokenizer=tokenizer,
                prompt=prompt_text,
                max_length=max_gen_len,
                temperature=temperature,
                top_k=top_k_val,
                sampling_method='top_k'
            )
            # Display Top-K
            full_text_top_k = display_prompt + generated_text_top_k.replace(prompt_text, "", 1) # Avoid repeating prompt
            st.markdown(full_text_top_k, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Comparison: Greedy Sampling")
             # Generate with Greedy
            generated_text_greedy = model.generate(
                tokenizer=tokenizer,
                prompt=prompt_text,
                max_length=max_gen_len,
                sampling_method='greedy' # Temperature and top_k are ignored for greedy
            )
            # Display Greedy
            full_text_greedy = display_prompt + generated_text_greedy.replace(prompt_text, "", 1)
            st.markdown(full_text_greedy, unsafe_allow_html=True)

else:
    st.error("Model or supporting files could not be loaded. Please ensure the checkpoint, tokenizer, and config files are in the specified directory and try again.")
    st.markdown(f"Expected files in `{current_model_dir}`:
"
                f"- `{DEFAULT_MODEL_NAME}` (Model checkpoint)
"
                f"- `{DEFAULT_TOKENIZER_NAME}` (Tokenizer vocabulary)
"
                f"- `{DEFAULT_CONFIG_NAME}` (Training configuration)")

st.sidebar.markdown("---")
st.sidebar.info("This app uses a character-level GPT-style model to generate movie plots based on a selected genre.") 