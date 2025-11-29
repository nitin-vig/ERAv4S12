import os
import sys
import warnings
import torch
import torch.nn.functional as F
import gradio as gr
import tiktoken
from model import GPT, GPTConfig

# Suppress asyncio cleanup warnings (harmless but noisy)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*asyncio.*")
import logging
logging.getLogger("asyncio").setLevel(logging.ERROR)

# Global variables
model = None
enc = None
device = 'cpu'
model_loaded = False

def load_model():
    """Load the model from checkpoint"""
    global model, enc, device, model_loaded
    
    # Skip if already loaded
    if model_loaded and model is not None:
        return "✅ Model already loaded!"
    
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = 'cpu'
    
    print(f"Loading model on device: {device}")
    
    # Initialize tokenizer
    if enc is None:
        enc = tiktoken.get_encoding('gpt2')
    
    # Load model and config from separate files
    model_path = 'gpt_model.pt'
    config_path = 'gpt_config.pt'
    
    # Check if files exist
    if not os.path.exists(model_path):
        return f"❌ Model file not found: {model_path}\n\nPlease upload gpt_model.pt to the Space root directory."
    if not os.path.exists(config_path):
        return f"❌ Config file not found: {config_path}\n\nPlease upload gpt_config.pt to the Space root directory."
    
    # Check file sizes (basic sanity check)
    model_size = os.path.getsize(model_path)
    config_size = os.path.getsize(config_path)
    if model_size == 0:
        return f"❌ Model file is empty: {model_path}"
    if config_size == 0:
        return f"❌ Config file is empty: {config_path}"
    
    print(f"Found model file: {model_path} ({model_size} bytes)")
    print(f"Found config file: {config_path} ({config_size} bytes)")
    
    try:
        # Suppress pickle/weights_only warnings - the config file is safe (it's your own model)
        import warnings
        with warnings.catch_warnings():
            # Suppress all warnings during loading
            warnings.simplefilter("ignore")
            
            # Load config (dataclass requires weights_only=False)
            # This is safe since it's your own model file
            config_data = torch.load(config_path, map_location=device, weights_only=False)
        
        # Handle both dict and dataclass formats
        if isinstance(config_data, dict):
            config = GPTConfig(**config_data)
        else:
            # It's already a GPTConfig dataclass
            config = config_data
        
        # Verify config is loaded correctly
        if config is None:
            raise ValueError("Config file is empty or corrupted")
        
        # Create model
        model = GPT(config)
        
        # Load state dict (use weights_only=True for state dict - it's just tensors)
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except Exception:
            # Fallback if weights_only fails
            state_dict = torch.load(model_path, map_location=device)
        
        if state_dict is None:
            raise ValueError("Model file is empty or corrupted")
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        model_loaded = True
        
        status_msg = "✅ Model loaded successfully from gpt_model.pt and gpt_config.pt!"
        if missing_keys:
            # Missing keys are usually buffers that get recreated, which is fine
            status_msg += f"\n⚠️ Note: {len(missing_keys)} buffer(s) will be recreated (this is normal)"
        if unexpected_keys:
            status_msg += f"\n⚠️ Note: {len(unexpected_keys)} unexpected key(s) ignored"
        
        return status_msg
    except FileNotFoundError as e:
        return f"❌ File not found: {str(e)}\n\nPlease make sure gpt_model.pt and gpt_config.pt are uploaded to the Space."
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error details: {error_trace}")  # Print to console for debugging
        return f"❌ Error loading model: {str(e)}\n\nCheck the console logs for full traceback."

def generate_text(prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    """Generate text from prompt"""
    global model, enc, device
    
    if model is None or enc is None:
        return "❌ Error: Model not loaded. Please load the model first."
    
    if not prompt.strip():
        return "❌ Error: Please provide a prompt."
    
    try:
        # Encode the prompt
        input_ids = enc.encode(prompt)
        
        # Truncate if too long
        max_input_length = model.config.block_size - max_new_tokens
        if len(input_ids) > max_input_length:
            input_ids = input_ids[-max_input_length:]
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Generate tokens
        generated_tokens = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits, _ = model(input_tensor)
                
                # Get logits for the last token
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    topk_logits, topk_indices = torch.topk(logits, top_k, dim=-1)
                    # Create a new tensor with -inf for non-top-k values
                    filtered_logits = torch.full_like(logits, float('-inf'))
                    filtered_logits.scatter_(1, topk_indices, topk_logits)
                    logits = filtered_logits
                
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                next_token_id = next_token.item()
                generated_tokens.append(next_token_id)
                
                # Update input tensor for next iteration (keep only last block_size tokens)
                if len(generated_tokens) > model.config.block_size:
                    generated_tokens = generated_tokens[-model.config.block_size:]
                input_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=device)
                
                # Stop if we hit the end token (50256 is <|endoftext|> in GPT-2)
                if next_token_id == 50256:
                    break
        
        # Decode the generated text
        generated_text = enc.decode(generated_tokens)
        
        # Return only the newly generated part
        original_length = len(input_ids)
        new_text = enc.decode(generated_tokens[original_length:])
        
        return new_text
    
    except Exception as e:
        return f"❌ Error during generation: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="GPT Text Generator") as demo:
    gr.Markdown("""
    # 🤖 GPT Text Generator
    
    A custom GPT model for text generation. Enter a prompt and generate text using the trained model.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=5,
                value="The future of artificial intelligence"
            )
            
            with gr.Accordion("Generation Parameters", open=False):
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10,
                    label="Max Tokens",
                    info="Maximum number of tokens to generate"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                    info="Higher values = more random, lower = more deterministic"
                )
                top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Top-K",
                    info="Sample from top K tokens (0 = no filtering)"
                )
            
            with gr.Row():
                generate_btn = gr.Button("🚀 Generate", variant="primary", scale=2)
                load_btn = gr.Button("🔄 Reload Model", variant="secondary", scale=1)
            
            status = gr.Textbox(
                label="Status",
                interactive=False,
                value="Click 'Reload Model' to load the model"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Generated Text")
            output = gr.Textbox(
                label="Output",
                lines=15,
                interactive=False
            )
    
    # Examples
    gr.Markdown("### Example Prompts")
    examples = gr.Examples(
        examples=[
            ["The future of artificial intelligence"],
            ["Once upon a time, in a distant galaxy"],
            ["The key to success is"],
            ["In the year 2050, technology will"],
        ],
        inputs=prompt_input
    )
    
    # Event handlers
    load_btn.click(
        fn=load_model,
        outputs=status
    )
    
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, max_tokens, temperature, top_k],
        outputs=output
    )
    
    # Auto-load model on startup
    demo.load(
        fn=load_model,
        outputs=status
    )

if __name__ == "__main__":
    demo.launch(share=False)

