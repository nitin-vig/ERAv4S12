# GPT-2 Training from Scratch

This project implements a GPT-2 model from scratch and trains it on custom text data. The implementation includes training scripts, model architecture, and a Hugging Face Space deployment for text generation.

## Project Overview

This repository contains:
- **Custom GPT-2 Architecture**: A PyTorch implementation of GPT-2 with proper weight initialization and residual scaling
- **Training Script**: Complete training pipeline with gradient accumulation, learning rate scheduling, and checkpointing
- **Hugging Face Space**: Interactive Gradio interface for text generation using the trained model

## Features

### Model Architecture
- **GPT-2 Base Configuration**: 12 layers, 12 attention heads, 768 embedding dimensions
- **Proper Weight Initialization**: Implements NanoGPT-style scaling for residual connections
- **Causal Self-Attention**: Multi-head attention with causal masking
- **GELU Activation**: Uses GELU with tanh approximation in MLP layers
- **Weight Sharing**: Token embedding weights are shared with the language model head

### Training Features
- **Gradient Accumulation**: Supports effective larger batch sizes through gradient accumulation
- **Learning Rate Scheduling**: Warmup followed by cosine decay
- **Checkpoint Saving**: Saves model checkpoints during training (supports Google Colab Drive integration)
- **Evaluation**: Periodic evaluation during training to monitor progress
- **Memory Optimization**: Batch size and sequence length tuned to prevent OOM errors

## Development History

Based on commit history, the project evolved through several iterations:

1. **Initial Setup** - Created basic project structure and README
2. **Batch Size Optimization** - Increased batch size and token length for better GPU utilization
3. **Memory Management** - Reduced batch size to fix Out-of-Memory (OOM) errors
4. **Gradient Accumulation** - Adjusted gradient accumulation steps and fixed loss calculation bugs
5. **Checkpoint System** - Added checkpoint saving functionality for model persistence

## Project Structure

```
ERAv4S12/
├── train_get2-8-init.py    # Main training script
├── main.py                  # Entry point
├── input.txt                # Training data (40,001 lines)
├── requirements.txt         # Python dependencies
├── gpt_model.pt             # Trained model weights
├── gpt_model_checkpoint.pt  # Full training checkpoint
├── gpt_config.pt            # Model configuration
└── hf_space/                # Hugging Face Space deployment
    ├── app.py              # Gradio interface
    ├── model.py            # Model architecture
    └── requirements.txt     # Space dependencies
```

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ERAv4S12
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements

- `torch>=2.0.0` - PyTorch for model training and inference
- `tiktoken>=0.5.0` - GPT-2 tokenizer
- `transformers>=4.30.0` - For loading pretrained weights (optional)

## Usage

### Training

Run the training script:
```bash
python train_get2-8-init.py
```

**Training Configuration:**
- Batch size: 32
- Sequence length: 128
- Max iterations: 5000
- Learning rate: 3e-4 (with warmup and cosine decay)
- Gradient accumulation: 2 steps (effective batch size: 64)
- Evaluation interval: Every 500 iterations

The script will:
1. Load and tokenize text from `input.txt`
2. Train the GPT model with the specified configuration
3. Save checkpoints to `./checkpoints/` (or Google Drive if running in Colab)
4. Save final model files: `gpt_model.pt`, `gpt_config.pt`, and `gpt_model_checkpoint.pt`

### Loading a Trained Model

```python
import torch
from train_get2-8-init import GPT, GPTConfig

# Load checkpoint
checkpoint = torch.load('gpt_model_checkpoint.pt')
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
```

### Hugging Face Space

The project includes a Gradio interface for interactive text generation. The Space is available at: [https://huggingface.co/spaces/nitinvig/gpt2_trained_on_colab](https://huggingface.co/spaces/nitinvig/gpt2_trained_on_colab)

To deploy locally:
```bash
cd hf_space
pip install -r requirements.txt
python app.py
```

## Model Configuration

The default GPT-2 configuration:
- **Block Size**: 1024 (maximum sequence length)
- **Vocabulary Size**: 50,257 tokens
- **Number of Layers**: 12
- **Number of Heads**: 12
- **Embedding Dimension**: 768

## Training Data

The model is trained on `input.txt` which contains 40,001 lines of text. The data is tokenized using GPT-2's BPE tokenizer via `tiktoken`.

## Key Implementation Details

### Residual Scaling
The implementation includes proper scaling for residual connections to address initialization issues:
- Output projections in attention and MLP layers use scaled initialization
- Scaling factor: `(2 * n_layer)^(-0.5)`

### Gradient Accumulation
- Loss is divided by `gradient_accumulation_steps` before backpropagation
- Gradients are accumulated over multiple steps before optimizer update
- Effective batch size = `batch_size * gradient_accumulation_steps`

### Learning Rate Schedule
- **Warmup**: Linear warmup for first 100 iterations
- **Decay**: Cosine decay from `learning_rate` to `min_lr` (3e-5)

## Device Support

The code automatically detects and uses the best available device:
- CUDA (if available)
- MPS (Apple Silicon)
- CPU (fallback)

## Notes

- The training script includes Google Colab integration for saving checkpoints to Google Drive
- Model checkpoints include optimizer state for resuming training
- The Hugging Face Space requires `gpt_model.pt` and `gpt_config.pt` files to be uploaded

## License

[Add your license here]

## Author

Nitin Vig
