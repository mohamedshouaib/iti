# LSTM Translation Project

This project implements a sequence-to-sequence (Seq2Seq) model with attention using Long Short-Term Memory (LSTM) networks for translating English sentences to German. The model is trained on the OPUS Books dataset and evaluated using the BLEU score.

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to build an accurate translation model using PyTorch and LSTM networks. The model consists of an encoder-decoder architecture with attention mechanism to improve translation quality.

## Dependencies
To run this project, you need the following dependencies:
- Python 3.11
- PyTorch 1.12 or higher
- spaCy
- Hugging Face Datasets
- sacrebleu
- tqdm

You can install the required dependencies using pip:
```bash
pip install torch torchvision torchaudio
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
pip install datasets sacrebleu tqdm
