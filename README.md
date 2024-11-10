# Multimodal Fusion Model for Offensive Content Classification

This project aims to build a multimodal fusion model that integrates both text and image inputs to classify content as offensive or non-offensive. By leveraging both Natural Language Processing (NLP) and Computer Vision techniques, this model provides a comprehensive understanding of text and visual data. Three different fusion techniques are explored: Early Fusion, Late Fusion, and Hybrid Fusion.

## Table of Contents

- [Overview](#overview)
- [Project Goals](#project-goals)
- [Dataset](#dataset)
- [Data Processing](#data-processing)
- [Fusion Techniques](#fusion-techniques)
- [Results and Analysis](#results-and-analysis)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project utilizes multimodal data (text and images) to develop a robust classification model that can determine the sentiment of meme content. The fusion model combines text features from a pre-trained BERT model and visual features from a pre-trained ResNet model to produce a binary classification output.

## Project Goals

- Combine text and image inputs for sentiment classification.
- Implement Early, Late, and Hybrid Fusion for feature representation.
- Leverage BERT for text and ResNet for image feature extraction.
- Compare the effectiveness of fusion techniques on accuracy.
- Deploy a Streamlit-based interface for real-time content classification.

## Dataset

- **Images**: 750 meme images containing both visual and textual content.
- **CSV File**: Accompanies each image with relevant text captions.
- **Data Splitting**: The data is divided into training, validation, and testing sets for effective model training and evaluation.

## Data Processing

- **Text Processing**: Text data is tokenized using BERT tokenizer, which provides input IDs and attention masks. The tokens are then passed through BERT to extract high-level text features.
- **Image Processing**: Images are resized to 224x224 pixels and normalized before being fed into ResNet to extract meaningful visual features.

## Fusion Techniques

1. **Early Fusion**: Combines raw features from both text and images early in the pipeline.
2. **Late Fusion**: Extracts features from each modality independently, then merges them for classification.
3. **Hybrid Fusion**: Integrates early and late fusion methods, enabling both independent and combined feature extraction for enhanced accuracy.

## Results and Analysis

- **Performance Comparison**: 
    - Hybrid Fusion achieved the highest accuracy (0.5771) and F1 Score (0.5733), outperforming both Early and Late Fusion models.
- **Effectiveness of Hybrid Fusion**: By integrating both early and late techniques, Hybrid Fusion captures a wider array of features, leading to more robust and accurate predictions.
- **Impact of Multimodality**: The use of both text and image modalities improves the model’s ability to classify content, leveraging unique strengths from each data type.

## Deployment Visuals
<img width="959" alt="mmal1" src="https://github.com/user-attachments/assets/38232462-e769-416f-afa4-93c789717890">
<img width="959" alt="mmal2" src="https://github.com/user-attachments/assets/f07c3a32-1234-4513-a705-27a8a9905966">


## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multimodal-fusion-classification.git
