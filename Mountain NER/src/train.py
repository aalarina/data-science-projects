# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

from transformers import AutoTokenizer

# Load the tokenizer corresponding to the pretrained BERT model.
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


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
