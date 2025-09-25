# Multi-Task Text Classification Model

A deep learning model that performs simultaneous classification of text across three different tasks: emotion detection, violence detection, and hate speech detection using a shared neural network architecture.

## Overview

This project implements a multi-task learning approach using TensorFlow/Keras to classify text into three categories:
- **Emotion Classification**: Classifies text into 6 emotional categories
- **Violence Detection**: Identifies 5 types of violence in text
- **Hate Speech Detection**: Detects offensive language and hate speech

## Features

- Multi-task neural network with shared layers for efficient learning
- Text preprocessing with stopword removal and tokenization
- Interactive classification interface using Jupyter widgets
- Comprehensive evaluation with confusion matrices
- Balanced dataset sampling for improved performance

## Model Architecture

The model uses a shared embedding and LSTM architecture:
- **Shared Embedding Layer**: 128-dimensional word embeddings
- **Shared LSTM Layer**: 64 units with return sequences
- **Global Average Pooling**: For feature extraction
- **Task-Specific Dense Layers**: Separate output layers for each task

```
Input Layers (3) → Shared Embedding → Shared LSTM → Pooling → Task-Specific Outputs (3)
```

## Dataset Requirements

The model expects three CSV files:
1. `text.csv` - Emotion dataset
2. `Train.csv` - Violence detection dataset  
3. `labeled_data.csv` - Hate speech dataset

### Expected Data Format
- **Emotion dataset**: `text` column with text data, `label` column with emotion labels (0-5)
- **Violence dataset**: `tweet` column with text data, `type` column with violence types
- **Hate speech dataset**: `tweet` column with text data, `class` column with hate speech labels

## Classification Categories

### Emotion Labels (6 classes)
- 0: Sadness
- 1: Joy  
- 2: Love
- 3: Anger
- 4: Fear
- 5: Surprise

### Violence Labels (5 classes)
- 0: Economic violence
- 1: Emotional violence
- 2: Harmful traditional practice
- 3: Physical violence
- 4: Sexual violence

### Hate Speech Labels (3 classes)
- 0: Offensive speech
- 1: Neither (normal speech)
- 2: Hate speech

## Installation

### Required Dependencies
```bash
pip install numpy pandas tensorflow scikit-learn nltk matplotlib seaborn ipywidgets
```

### NLTK Data
The script automatically downloads required NLTK data:
- stopwords
- punkt
- punkt_tab

## Usage

### 1. Data Preparation
Ensure your CSV files are in the correct format and accessible at the specified paths in the script.

### 2. Training the Model
```python
# Run the complete script to:
# - Load and preprocess datasets
# - Balance the datasets (12K samples each)
# - Train the multi-task model
# - Generate confusion matrices
```

### 3. Text Classification
Use the interactive widget or the `classify_text()` function:

```python
major_label, sub_label = classify_text("Your input text here")
print(f"Major Category: {major_label}")
print(f"Specific Classification: {sub_label}")
```

### 4. Interactive Interface
The script provides a Jupyter widget interface for real-time text classification.

## Model Performance

The model is evaluated using:
- Confusion matrices for each task
- Individual accuracy metrics per task
- Visual heatmaps for performance analysis

## Data Preprocessing

1. **Column Standardization**: Renames columns to consistent format (`text`, `label`)
2. **Data Cleaning**: Removes unnecessary columns and handles missing values
3. **Dataset Balancing**: Samples equal amounts from each dataset (12K samples)
4. **Label Encoding**: Converts categorical labels to numerical format
5. **Stopword Removal**: Removes common English stopwords
6. **Tokenization**: Converts text to numerical sequences
7. **Padding**: Ensures uniform sequence length (max 50 tokens)

## Model Configuration

- **Optimizer**: Adam
- **Loss Function**: Sparse categorical crossentropy for all tasks
- **Metrics**: Accuracy for each task
- **Epochs**: 10
- **Batch Size**: 4
- **Max Sequence Length**: 50 tokens

## File Structure

```
project/
├── untitled19.py          # Main script
├── text.csv              # Emotion dataset
├── Train.csv             # Violence dataset
├── labeled_data.csv      # Hate speech dataset
└── README.md            # This file
```

## Key Functions

- `remove_stopwords(text)`: Removes English stopwords from text
- `plot_cm(true, pred, title, labels)`: Creates confusion matrix visualizations
- `classify_text(input_text)`: Classifies new text input
- `on_button_click(b)`: Handles interactive widget events

## Notes

- The model uses shared layers to learn common text representations across tasks
- Dataset balancing ensures fair representation across all categories
- The interactive interface allows real-time testing of the model
- Confusion matrices help identify model strengths and weaknesses per task

## Future Improvements

- Implement cross-validation for more robust evaluation
- Add more sophisticated text preprocessing (lemmatization, spell checking)
- Experiment with transformer-based architectures
- Include confidence scores in predictions
- Add model saving/loading functionality

## Troubleshooting

**Common Issues:**
- Ensure all required CSV files are present and accessible
- Verify column names match expected format
- Check that datasets contain sufficient samples for balancing
- Install all required dependencies before running

For optimal performance, run on a system with sufficient RAM and consider using GPU acceleration for faster training.
