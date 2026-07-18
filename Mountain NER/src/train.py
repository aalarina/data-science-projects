"""
Mountain Named Entity Recognition

This script fine-tunes a BERT model for the Mountain Named Entity Recognition
(NER) task using the Hugging Face Transformers library.
"""

# ==================================================
# Imports
# ==================================================

import numpy as np
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)


# ==================================================
# Configuration
# ==================================================

MODEL_NAME = "bert-base-cased"

OUTPUT_DIR = "./mountain_ner"

MODEL_DIR = "mountain_ner_model"

LEARNING_RATE = 2e-5

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

NUM_EPOCHS = 5

WEIGHT_DECAY = 0.01


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

# Load the tokenizer corresponding to the pretrained BERT model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# --------------------------------------------------
# Tokenization and label alignment
# --------------------------------------------------

def tokenize_and_align_labels(examples):
    """
    Tokenize the input sentences and align BIO labels with BERT subword tokens.

    Special tokens ([CLS], [SEP]) and continuation subword pieces receive
    the label -100 so they are ignored during loss computation.
    """

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    labels = []

    for i, label in enumerate(examples["labels"]):

        # Map each tokenized piece to its original word.
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None

        label_ids = []

        for word_idx in word_ids:

            # Ignore special tokens.
            if word_idx is None:
                label_ids.append(-100)

            # Assign the label only to the first subword.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            # Ignore remaining subword pieces.
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


# --------------------------------------------------
# Tokenize the dataset
# --------------------------------------------------

tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
)

print("Tokenization completed successfully.")


# --------------------------------------------------
# Load model
# --------------------------------------------------

# Load the pretrained BERT model for token classification.
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)


# --------------------------------------------------
# Data collator
# --------------------------------------------------

# Dynamically pad batches during training.
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer
)


# --------------------------------------------------
# Evaluation metric
# --------------------------------------------------

# Load the seqeval metric for NER evaluation.
metric = evaluate.load("seqeval")


def compute_metrics(eval_pred):
    """
    Compute Precision, Recall, F1-score and Accuracy
    using the seqeval evaluation library.
    """

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):

        current_predictions = []
        current_labels = []

        for pred, lab in zip(prediction, label):

            # Ignore labels assigned to special tokens
            # and subword continuations.
            if lab != -100:

                current_predictions.append(
                    id2label[pred]
                )

                current_labels.append(
                    id2label[lab]
                )

        true_predictions.append(current_predictions)
        true_labels.append(current_labels)

    results = metric.compute(
        predictions=true_predictions,
        references=true_labels,
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # Evaluate and save the model after every epoch.
    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=5,

    weight_decay=0.01,

    # Restore the checkpoint with the best validation score.
    load_best_model_at_end=True,

    logging_steps=10,

    report_to="none",
)


# --------------------------------------------------
# Trainer
# --------------------------------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset["train"],

    eval_dataset=tokenized_dataset["validation"],

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


# --------------------------------------------------
# Train the model
# --------------------------------------------------

trainer.train()


# --------------------------------------------------
# Save the trained model
# --------------------------------------------------

trainer.save_model("mountain_ner_model")
tokenizer.save_pretrained("mountain_ner_model")

print("Model successfully saved.")


# --------------------------------------------------
# Optional: create a ZIP archive (Google Colab)
# --------------------------------------------------

import shutil

shutil.make_archive(
    "mountain_ner_model",
    "zip",
    "mountain_ner_model",
)

try:
    from google.colab import files

    files.download("mountain_ner_model.zip")

except ImportError:
    pass
