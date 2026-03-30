# Transformer-Based Sentiment Classification using DistilBERT

## Overview
This project implements a transformer-based sentiment classification model using a pre-trained DistilBERT model. The goal is to explore how modern Natural Language Processing (NLP) techniques can capture contextual meaning in text more effectively than traditional machine learning approaches.

## Research Question
How effectively can transformer-based models classify sentiment in text compared to traditional machine learning methods in terms of accuracy and contextual understanding?

## Dataset
The dataset used is the IMDB movie review dataset, accessed via the Hugging Face datasets library. Reviews are labeled as either positive or negative.

For this project, a subset of the test set is used for evaluation.

## Methodology
- Pre-trained DistilBERT model via Hugging Face Transformers
- Tokenization and inference using the `pipeline` API
- Truncation applied to handle long sequences
- Predictions converted into binary labels
- Evaluation using accuracy metric

## Results
The transformer-based model achieved an accuracy of approximately 0.94 on the evaluation subset of the IMDB dataset.

Sample predictions indicate that the model can correctly classify both positive and negative reviews with high confidence. The results demonstrate strong performance compared to traditional approaches such as TF-IDF with Logistic Regression.

## Interpretation
The results show that transformer-based models are highly effective for sentiment classification tasks. By leveraging contextual embeddings, the model captures nuances in language that simpler models often miss.

Compared to traditional machine learning methods, the transformer model provides improved accuracy and more reliable predictions, particularly for complex or context-dependent text.

## Limitations
- Evaluation performed on a subset of the dataset
- No fine-tuning of the model was performed
- Performance may vary on longer or highly ambiguous text
- Higher computational cost compared to traditional models

## Future Work
- Fine-tune the transformer model on the full dataset
- Compare with other transformer architectures (e.g., BERT, RoBERTa)
- Perform error analysis on misclassified examples
- Apply the model to real-world datasets

## Requirements
Install dependencies using:

```bash
pip install transformers datasets torch scikit-learn pandas
