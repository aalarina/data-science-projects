# Mountain Named Entity Recognition

## Project Overview

This project implements a Named Entity Recognition (NER) system for detecting mountain names in English text.

The solution covers the complete machine learning pipeline, including:
- automatic dataset creation;
- BIO annotation generation;
- fine-tuning a pretrained BERT model;
- model evaluation;
- inference and visualization of predictions.

The final model was fine-tuned using the Hugging Face Transformers library on a custom dataset created specifically for this task.

---

## Solution Explanation 

The project was implemented in three main stages.

### 1. Dataset Creation

Since no suitable annotated dataset for mountain name recognition was available, a custom dataset was created.

The dataset creation pipeline consists of the following steps:

- mountain names are automatically collected from the Wikipedia page *List of mountain peaks by prominence*;
- extracted names are cleaned by removing formatting symbols, splitting alternative names and eliminating duplicates;
- aliases are generated for mountains (for example, *Mount Everest* → *Everest*);
- synthetic English sentences are generated using ChatGPT based on predefined requirements regarding writing style, sentence length and entity distribution;
- BIO annotations are created automatically by matching mountain names within each sentence;
- the dataset is split into training, validation and test sets.

### 2. Model Training

The model is based on the pretrained **bert-base-cased** model from Hugging Face.

The training pipeline includes:

- tokenization using the BERT tokenizer;
- alignment of BIO labels with WordPiece tokens;
- fine-tuning for token classification;
- evaluation using Precision, Recall, F1-score and Accuracy.

**Why BERT?**
The bert-base-cased model was selected because it provides strong baseline performance for token classification tasks while remaining lightweight enough for efficient fine-tuning within the scope of this project.

### 3. Inference

During inference, the trained model predicts mountain names in arbitrary English text.

An additional post-processing step merges WordPiece fragments (for example, *E* + *##tna* → *Etna*) before displaying the final predictions.

The project also provides HTML visualization that highlights detected mountain names together with confidence scores.

---

## Project Structure

```text
Mountain-NER/
│
├── notebooks/
│   ├── dataset_creation.ipynb   # dataset generation and BIO annotation
│   └── demo_inference.ipynb     # inference examples
│
├── src/
│   ├── train.py                 # model fine-tuning
│   └── inference.py             # prediction utilities
│
├── mountain_ner_dataset.csv     # generated dataset
├── README.md
└── requirements.txt
```

---

## Model Weights

The fine-tuned model weights are available on Hugging Face:

https://huggingface.co/aalarina/mountain_ner_model

---

## Project Setup

Clone the repository.

```bash
git clone <repository-url>
cd Mountain-NER
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Dataset Creation

Run the notebook:

```
dataset_creation.ipynb
```

The notebook automatically:
- downloads mountain names;
- generates aliases;
- creates synthetic sentences;
- generates BIO annotations;
- exports the final dataset.

### Model Training

Run:

```bash
python train.py
```

The trained model is saved into:

```
mountain_ner_model/
```

### Inference

Run:

```bash
python inference.py
```

The project provides two inference functions:

- `predict(text)` — prints detected mountain names together with confidence scores;
- `highlight_mountains(text)` — displays the input text with detected mountain names highlighted.

---

## Evaluation Results

The model was evaluated on the held-out test set.

| Metric | Score |
|---------|------:|
| Precision | 0.850 |
| Recall | 0.809524 |
| F1-score | 0.829268 |
| Accuracy | 0.988263 |

---

## Limitations

Although the model achieves good performance on the test set, several limitations remain.

- The dataset is relatively small and contains fewer than 400 annotated sentences.
- Most training examples are synthetic rather than collected from real-world corpora.
- The model was trained only on English text.
- Recognition quality depends on the diversity of mountain names included in the training dataset.
- The model is specialized for mountain name recognition and is not intended for general-purpose Named Entity Recognition.

---

## Future Improvements

Several directions could further improve the project.

- Expand the dataset with real-world text collected from sources such as Wikipedia, travel blogs and news articles.
- Increase the diversity of sentence structures and mountain names.
- Experiment with more recent transformer architectures such as DeBERTa or RoBERTa.
- Perform systematic hyperparameter optimization.
- Extend the model to support multilingual mountain name recognition.

