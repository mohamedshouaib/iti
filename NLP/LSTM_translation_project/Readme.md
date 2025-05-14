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
```

## Dataset
The project uses the OPUS Books dataset, which contains parallel texts in English and German. The dataset is loaded using the Hugging Face Datasets library.


## Model Architecture
The model consists of:
Encoder: An LSTM network that processes the input sequence.
Attention Mechanism: Allows the decoder to focus on different parts of the input sequence.
Decoder: An LSTM network that generates the translated sequence.


## Training
The model is trained using the Adam optimizer and a learning rate scheduler. The training process involves:
- Tokenizing and preprocessing the dataset.
- Creating data loaders for training and validation.
- Training the model for a specified number of epochs.
- Saving the best model based on validation loss and BLEU score.


## Evaluation
The model's performance is evaluated using the BLEU score, which measures the similarity between the generated translations and reference translations.


## Usage
To run the training and evaluation process, follow these steps:
Clone the repository:
```
git clone git clone https://github.com/mohamedshouaib/iti/tree/main/NLP/LSTM_translation_project
cd lstm-translation-project
```
Install the dependencies:
```
pip install -r requirements.txt
```
Run the training script:
```
python train.py
```


## Results
The model achieves a BLEU score of approximately 16.25 on the validation set. Sample translations are printed during the evaluation process.


## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.


## License
This project is licensed under the MIT License - see the LICENSE file for details.
```
### Additional Tips
- **Screenshots**: Include screenshots of the training process or sample translations to make the README more engaging.
- **Example Outputs**: Provide examples of translated sentences to demonstrate the model's capabilities.
- **Known Issues**: List any known issues or limitations of the current implementation.
- **Future Work**: Suggest potential improvements or future work that could be done on the project.

This README file should provide a clear overview of your project, making it easier for others to understand and contribute to your work.
```



