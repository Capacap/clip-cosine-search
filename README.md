# CLIP cosine search

Bidirectional image-text search on Flickr8K using CLIP embeddings and cosine similarity in numpy. No vector DB, no LangChain.

> I built this to understand vector retrieval from first principles rather than wrapping a library. The similarity search is a normalized dot product in numpy. The notebooks show the math directly instead of hiding it behind a framework.

## Overview

OpenAI's CLIP model produces 512-dimensional embeddings for both images and text in a shared space. This project uses those embeddings to search across modalities: type a sentence to find matching images, or upload an image to find matching captions.

### Key Components

- **Data Processing**: Loads the Flickr8K dataset and generates normalized CLIP embeddings for every image and caption
- **Search Engine**: Computes similarity as a dot product over normalized embeddings
- **Web Interface**: Gradio app with tabs for text-to-image and image-to-text search
- **Storage**: HDF5 with gzip compression for the embedding store

## Features

### Text to Image Search
Find images that match natural language descriptions. The system searches through 8,091 images from the Flickr8K dataset.

### Image to Text Search  
Upload an image to find semantically similar text descriptions from the dataset.

### Web Interface
Gradio interface with a tab for each search direction. Result count is adjustable, and similarity scores are shown next to each result.

## Technical Specifications

- **Model**: OpenAI CLIP-ViT-Base-Patch32
- **Dataset**: Flickr8K (40,455 image-caption pairs)
- **Embedding Dimension**: 512
- **Storage Format**: HDF5 with gzip compression
- **Similarity Metric**: Cosine similarity (via normalized dot product)
- **Interface Framework**: Gradio

## Installation

```bash
# Clone repository
git clone <repository-url>
cd clip-cosine-search

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir -p data
```

## Usage

### 1. Data Preparation
```bash
jupyter notebook notebooks/part01_data_preparation.ipynb
```
Downloads Flickr8K dataset, generates CLIP embeddings, and stores them in HDF5 format.

### 2. Search Functionality Testing
```bash
jupyter notebook notebooks/part02_search_functionality.ipynb
```
Walks through the search code and compares result filtering approaches.

### 3. Launch Web Interface
```bash
jupyter notebook notebooks/part03_multimodal_interface.ipynb
```
Launches the Gradio web interface.

## Technical Details

CLIP embeddings are L2-normalized once at index time, which reduces cosine similarity to a single dot product per query:

```
similarity(query, target) = normalized_query · normalized_target
```

## Project Structure

```
clip-cosine-search/
├── notebooks/
│   ├── part01_data_preparation.ipynb      # Dataset loading and embedding generation
│   ├── part02_search_functionality.ipynb  # Search algorithm implementation and testing
│   └── part03_multimodal_interface.ipynb  # Web interface
├── data/
│   ├── flickr8k_embeddings.h5            # Stored embeddings
│   └── datasets/
│       └── adityajn105/flickr8k/
│           ├── Images/                     # Dataset images
│           └── captions.txt               # Image captions
├── requirements.txt                        # Python dependencies
└── README.md                              # Documentation
```

## Performance

- **Embedding Generation**: ~4.46 batches/second (RTX 3060 Ti, batch_size=32)
- **Search Speed**: Sub-second response for 40K+ embeddings
- **Storage**: HDF5 compression reduces storage by ~60%
- **Requirements**: CUDA-compatible GPU recommended

## Dependencies

Core libraries:
- `torch` - PyTorch framework
- `transformers` - Hugging Face transformers library
- `h5py` - HDF5 file handling
- `gradio` - Web interface
- `numpy` - Array operations
- `Pillow` - Image processing

See `requirements.txt` for the full dependency list with versions.
