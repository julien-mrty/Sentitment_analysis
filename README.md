# Sentiment Analysis Project

## Overview

This project focuses on classifying Amazon book reviews by predicting the critic's rating based on the review's title, comment, and helpfulness score. The objective is to develop a model that accurately estimates the rating a reviewer would assign, using natural language processing techniques.

## Features

- **Data Processing**: Cleans and preprocesses Amazon book review data, including handling missing values and encoding categorical variables.
- **Model Training**: Implements and fine-tunes BERT and DistilBERT models on the processed dataset to predict review ratings.
- **Evaluation**: Assesses model performance using metrics such as accuracy, precision, recall, and F1-score.


## Usage

1. **Prepare the dataset**: Ensure you have the Amazon book reviews dataset in the appropriate format.
2. **Preprocess the data**: Use the provided data processing scripts to clean and prepare the data for modeling.
3. **Train the model**: Run the training script to fine-tune the BERT or DistilBERT model on your dataset.
4. **Evaluate the model**: Use the evaluation scripts to assess the model's performance on a test set.

## Acknowledgements

- [BERT](https://github.com/google-research/bert)
- [DistilBERT](https://github.com/huggingface/transformers)
- [Amazon Customer Reviews Dataset](https://registry.opendata.aws/amazon-reviews/)
