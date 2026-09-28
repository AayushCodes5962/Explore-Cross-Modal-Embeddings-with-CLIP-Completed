# Explore Cross-Modal Embeddings with CLIP

A practical exploration of CLIP, a vision-language model developed by OpenAI, using Hugging Face Transformers.

This project demonstrates how CLIP maps images and text into a shared embedding space, allowing semantic relationships between different modalities to be measured through similarity scores.

The notebook loads an image, processes multiple text descriptions, calculates image-text similarity scores, converts the scores into probabilities, and visualizes the results.

## Overview

Traditional machine learning models often focus on a single modality such as images or text.

CLIP is designed to connect these modalities by learning representations for both images and text in a shared latent space.

This allows the model to determine how well a text description corresponds to a given image.

The workflow implemented in this project is:

```text
Input Image
     |
     v
CLIP Image Encoder
     |
     v
Image Embedding
     |
     |
     +----------------------+
                            |
                            v
                     Similarity Score
                            ^
                            |
     +----------------------+
     |
     v
Text Captions
     |
     v
CLIP Text Encoder
     |
     v
Text Embeddings
```

## Objectives

The main objectives of this project are to:

* Understand CLIP and cross-modal embeddings
* Load a pre-trained CLIP model
* Process image and text inputs simultaneously
* Generate image and text embeddings
* Calculate image-text similarity scores
* Convert similarity scores into probability distributions
* Identify the best matching caption
* Visualize image-text similarity
* Understand semantic alignment between vision and language
* Explore potential applications of multimodal embeddings

## Model

The project uses the following pre-trained CLIP model:

```text
openai/clip-vit-base-patch32
```

The model is loaded using Hugging Face Transformers:

```python
from transformers import CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
```

The CLIP architecture contains separate encoders for images and text.

These encoders transform their respective inputs into representations that can be compared within a shared embedding space.

## Technologies Used

| Technology                | Purpose                               |
| ------------------------- | ------------------------------------- |
| Python                    | Core programming language             |
| PyTorch                   | Tensor operations and model inference |
| Hugging Face Transformers | CLIP model and processor              |
| Pillow                    | Image loading and processing          |
| Matplotlib                | Visualization                         |
| NumPy                     | Numerical operations                  |
| Requests                  | Downloading the example image         |
| Jupyter Notebook          | Development and experimentation       |

## Project Workflow

```text
Load CLIP Model
       |
       v
Load Example Image
       |
       v
Define Text Captions
       |
       v
Process Image + Text
       |
       v
Generate CLIP Representations
       |
       v
Calculate Similarity Scores
       |
       v
Apply Softmax
       |
       v
Identify Best Matching Caption
       |
       v
Visualize Results
```

# Task 1: Load and Inspect the CLIP Model

The first part of the notebook initializes the CLIP model and processor.

The processor handles:

* Text tokenization
* Image preprocessing
* Tensor formatting
* Padding
* Batch preparation

The notebook uses an example image downloaded from the web.

The image is converted to RGB format before being processed.

## Text Captions

Multiple captions are defined to test the relationship between the image and different descriptions.

The notebook uses captions such as:

```text
A laptop computer on a desk
A dog playing in a park
A plate of delicious food
A person riding a bicycle
A laptop with a keyboard
```

These captions provide different levels of relevance to the input image.

This allows CLIP's semantic matching capability to be explored.

# Task 2: Encode Image and Text

The CLIP processor receives both the image and captions:

```python
inputs = processor(
    text=captions,
    images=image,
    return_tensors="pt",
    padding=True
)
```

The resulting input contains tensors required by the model.

The notebook examines:

```text
Input IDs
Attention Mask
Pixel Values
```

## Input IDs

`input_ids` contain the tokenized representation of the text captions.

These tokens are passed to the CLIP text encoder.

## Attention Mask

The attention mask identifies valid tokens within the padded text sequences.

## Pixel Values

`pixel_values` contain the preprocessed image representation used by the CLIP vision encoder.

## Why Joint Processing Matters

Processing the image and captions through the same CLIP processor ensures that both modalities are correctly formatted for model inference.

The resulting representations can then be compared to measure their semantic similarity.

# Task 3: Calculate Similarity Scores

The model performs inference using:

```python
with torch.no_grad():
    outputs = model(**inputs)
```

The notebook extracts:

```python
outputs.logits_per_image
```

These values represent the similarity between the input image and each text caption.

A higher similarity score indicates stronger semantic alignment between the image and caption.

## Probability Distribution

The similarity scores are converted into probabilities using softmax:

```python
probabilities = torch.softmax(similarity_scores, dim=0)
```

This produces a normalized distribution across the candidate captions.

The caption with the highest probability is selected as the best matching caption.

## Example Workflow

```text
Image
  |
  v
Compare with Caption 1
Compare with Caption 2
Compare with Caption 3
Compare with Caption 4
Compare with Caption 5
  |
  v
Similarity Scores
  |
  v
Softmax
  |
  v
Probability Distribution
  |
  v
Best Matching Caption
```

# Visualization

The notebook visualizes the similarity results using Matplotlib.

A horizontal bar chart is used to compare the probabilities associated with each caption.

The notebook also creates a combined visualization containing:

* Input image
* Caption similarity results

This makes it easier to understand which text descriptions are most strongly aligned with the visual content.

# Cross-Modal Embeddings

One of the main concepts explored in this project is cross-modal embedding.

CLIP maps both images and text into a shared latent representation.

Conceptually:

```text
Image
  |
  v
Image Encoder
  |
  v
Image Embedding
       |
       | Similarity
       |
       v
Text Embedding
  ^
  |
Text Encoder
  ^
  |
Text
```

Because the representations exist in the same embedding space, semantic similarity between images and text can be calculated.

This enables applications where a user can search or interact across different modalities.

# Manual Cosine Similarity

The notebook also extracts image and text embeddings separately:

```python
text_embeddings = model.get_text_features(...)
image_embeddings = model.get_image_features(...)
```

The embeddings are normalized before calculating similarity.

The cosine similarity can then be computed using:

```python
manual_similarities = torch.matmul(
    image_embeddings,
    text_embeddings.T
)
```

This provides an additional way to understand how CLIP compares the two modalities.

## Embedding Dimensions

The notebook examines:

* Text embedding shape
* Image embedding shape
* Embedding dimension
* Manual cosine similarities

This helps demonstrate the numerical representation used by CLIP internally.

# Key Concepts

## CLIP

CLIP is a vision-language model that learns relationships between images and natural language.

It can be used for tasks involving both visual and textual information.

## Cross-Modal Embeddings

Cross-modal embeddings represent different types of data in a common vector space.

In this project:

```text
Images → Image Embeddings
Text → Text Embeddings
```

The resulting representations can then be compared.

## Similarity Score

Similarity scores indicate how strongly an image and text description are related according to the model.

Higher scores indicate stronger alignment.

## Softmax

Softmax converts raw similarity scores into a normalized probability distribution.

This makes it easier to compare candidate captions.

## Zero-Shot Matching

CLIP can compare an image with text descriptions without requiring a task-specific classifier to be trained for those exact labels.

This provides the basis for zero-shot image-text matching.

# Applications

Cross-modal embeddings can be applied to several real-world AI applications.

### Image Search

Users can search for images using natural language descriptions.

```text
Text Query
    |
    v
Text Embedding
    |
    v
Compare with Image Embeddings
    |
    v
Relevant Images
```

### Zero-Shot Image Classification

Images can be compared against candidate class descriptions without training a dedicated classifier for every class.

### Content Recommendation

Images and text can be represented in the same embedding space to improve multimodal recommendation systems.

### Visual Question Answering

Cross-modal representations can form part of systems that combine visual and language information.

### Image-Text Retrieval

The same embedding approach can be used for both:

* Text-to-image retrieval
* Image-to-text retrieval

### Multimodal Search

Users can potentially search across text and visual content using a unified representation.

# Learning Outcomes

This project provides practical experience with:

* CLIP
* Vision-language models
* Cross-modal embeddings
* Image embeddings
* Text embeddings
* Similarity measurement
* Cosine similarity
* Softmax normalization
* Zero-shot image-text matching
* Hugging Face Transformers
* Multimodal AI
* Visualization of model outputs

# Installation

Install the required dependencies:

```bash
pip install transformers torch torchvision pillow matplotlib requests
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-link>
```

### 2. Navigate to the Project

```bash
cd <repository-name>
```

### 3. Install Dependencies

```bash
pip install transformers torch torchvision pillow matplotlib requests
```

### 4. Open the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
clip.ipynb
```

### 5. Run the Notebook

Run the cells sequentially.

The notebook will:

1. Install the required libraries
2. Load the CLIP model and processor
3. Download and display an example image
4. Define multiple captions
5. Encode image and text inputs
6. Calculate similarity scores
7. Convert scores into probabilities
8. Identify the best matching caption
9. Visualize the results
10. Examine cross-modal embeddings
11. Calculate manual cosine similarities

# Project Structure

```text
CLIP-Cross-Modal-Embeddings/
|
├── clip.ipynb
└── README.md
```

# Limitations

This project is an educational exploration of CLIP and does not implement a production-scale multimodal system.

The notebook:

* Uses a single example image
* Uses a small set of manually created captions
* Does not perform large-scale retrieval
* Does not train or fine-tune CLIP
* Does not evaluate CLIP across a formal benchmark dataset
* Does not implement a production search system

The similarity results should therefore be interpreted as an experiment demonstrating the concept rather than as a comprehensive evaluation of CLIP.

# Future Improvements

The project can be extended by:

* Testing CLIP on a larger image dataset
* Building a text-to-image search engine
* Implementing image-to-image retrieval
* Creating a zero-shot image classifier
* Comparing multiple CLIP models
* Fine-tuning CLIP on a custom dataset
* Building a multimodal recommendation system
* Creating an interactive Streamlit application
* Storing image embeddings in a vector database
* Combining CLIP with FAISS for large-scale image retrieval
* Exploring other vision-language models

# Conclusion

This project provides a practical introduction to cross-modal embeddings using CLIP.

By encoding images and text into a shared representation space, CLIP can measure the semantic relationship between visual content and natural language descriptions.

The notebook demonstrates the complete process from model initialization and input processing to similarity calculation, probability normalization, visualization, and manual embedding comparison.

These concepts provide a foundation for building more advanced multimodal applications such as image search, zero-shot classification, recommendation systems, and vision-language applications.

# Author

Aayush Kumar
