# SentimentSeer-LSTM

**SentimentSeer-LSTM** is a repository featuring a sentiment analysis project that leverages Long Short-Term Memory (LSTM) networks. This notebook demonstrates the entire workflow of preparing text data, building an LSTM-based sentiment classification model, training it, and evaluating its performance.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Data Preparation](#data-preparation)
4. [Model Architecture](#model-architecture)
5. [Training the Model](#training-the-model)
6. [Model Evaluation](#model-evaluation)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

## Overview

SentimentSeer-LSTM provides a comprehensive implementation of sentiment analysis using LSTM networks. The primary components of this project include:

- **Data Preparation**: Includes downloading and preprocessing text data.
- **Model Architecture**: Describes the LSTM network used for sentiment classification.
- **Training**: Outlines the process of training the LSTM model on the preprocessed data.
- **Evaluation**: Details how to evaluate the model’s performance on test data.
- **Inference**: Shows how to use the trained model for making predictions on new text data.

## Getting Started

To run the notebook, ensure that you have the following dependencies installed:

- **Python** (>= 3.6)
- **PyTorch** (>= 1.0)
- **NumPy**
- **Collections**
- **String**

Install the required Python packages using pip:

```sh
pip install numpy torch
```

## Data Preparation

The dataset used consists of movie reviews and their associated sentiment labels. Data is first downloaded and then preprocessed, which includes:

- **Reading Text Files**: The reviews and labels are loaded from text files.
- **Text Normalization**: Converting all text to lowercase and removing punctuation.
- **Tokenization**: Splitting text into words and converting words to integer tokens.
- **Padding**: Ensuring that all sequences are of uniform length for model input.

## Model Architecture

The sentiment analysis model is built using Long Short-Term Memory (LSTM) networks. The model architecture consists of:

- **Embedding Layer**: Transforms word indices into dense vectors of fixed size.
- **LSTM Layer**: Processes the sequences to capture temporal dependencies.
- **Dropout Layer**: Regularizes the model to prevent overfitting.
- **Fully Connected Layer**: Maps LSTM outputs to sentiment classes.
- **Sigmoid Activation**: Produces the final probability of the review being positive.

![Rnn_Arch](/images/RNN_arch.png)

## Training the Model

The model is trained using the following steps:

1. **Data Loading**: Data is split into training, validation, and test sets. Dataloaders are used to handle batching and shuffling of the training data.
2. **Loss Function**: Binary Cross Entropy Loss is used to measure the error in predictions.
3. **Optimizer**: Adam optimizer is used for updating the model’s parameters.
4. **Epochs**: The model is trained over a set number of epochs, with validation loss being monitored to prevent overfitting.

## Model Evaluation

After training, the model is evaluated on a separate test set to determine its performance. Key metrics include:

- **Test Loss**: Average loss on the test data => 0.477
- **Test Accuracy**: Proportion of correctly classified reviews. => 0.803 (trained for 4 epochs)

The evaluation helps in understanding how well the model generalizes to unseen data.

## Usage

To use the trained model for predicting the sentiment of new reviews:

1. **Tokenize and Pad Review**: Convert the review into tokenized integers and pad it to the required sequence length.
2. **Predict**: Pass the processed review through the model to get the sentiment prediction.

Examples of how to use the model for prediction are provided in the notebook.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the existing style and includes relevant tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
