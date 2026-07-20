"""
Mountain Named Entity Recognition

This script loads the trained model and performs inference
on user-provided text.
"""

# ==================================================
# Imports
# ==================================================

from transformers import pipeline
from IPython.display import display, HTML


# ==================================================
# Load trained model
# ==================================================

ner = pipeline(
    "token-classification",
    model="mountain_ner_model",
    tokenizer="mountain_ner_model",
    aggregation_strategy="simple",
)


# ==================================================
# Merge WordPiece tokens
# ==================================================

def merge_subwords(predictions):
    """
    Merge WordPiece fragments produced by the tokenizer.

    Example:
        E + ##tna -> Etna
    """

    if not predictions:
        return predictions

    merged = []

    for pred in predictions:

        # Merge continuation tokens with the previous prediction
        if pred["word"].startswith("##") and merged:

            merged[-1]["word"] += pred["word"][2:]
            merged[-1]["end"] = pred["end"]

            # Update the confidence score using the mean value
            merged[-1]["score"] = (
                merged[-1]["score"] + pred["score"]
            ) / 2

        else:

            merged.append(pred.copy())

    return merged


# ==================================================
# Prediction
# ==================================================

def predict(text):
    """
    Detect mountain names in the input text.
    """

    # Run inference and merge fragmented WordPiece predictions
    predictions = merge_subwords(ner(text))

    if not predictions:
        print("No mountains found.")
        return

    print("Detected mountains:")

    # Print each detected mountain together with its confidence score
    for entity in predictions:

        print(
            f"- {entity['word']} "
            f"(confidence={entity['score']:.3f})"
        )


# ==================================================
# HTML visualization
# ==================================================

def highlight_mountains(text):
    """
    Highlight detected mountain names in HTML.
    """

    # Run inference and merge fragmented WordPiece predictions
    predictions = merge_subwords(ner(text))

    html_text = text

    # Process entities in reverse order to preserve character indices
    for entity in sorted(predictions, key=lambda x: x["start"], reverse=True):

        start = entity["start"]
        end = entity["end"]

        # Extract the original text span
        word = text[start:end]

        highlighted = f"""
        <span style="
            background:#FFD54F;
            padding:2px 5px;
            border-radius:4px;
            font-weight:bold;
        ">
            {word}
        </span>
        <span style="color:#666;font-size:13px;">
            ({entity['score']:.3f})
        </span>
        """

        # Replace the original entity with the highlighted HTML snippet
        html_text = (
            html_text[:start]
            + highlighted
            + html_text[end:]
        )

    # Display the formatted HTML output
    display(HTML(f"""
    <div style="
        font-size:18px;
        line-height:1.8;
        border:1px solid #ddd;
        border-radius:10px;
        padding:20px;
        background:white;
    ">
        {html_text}
    </div>
    """))


# ==================================================
# Example
# ==================================================

if __name__ == "__main__":
    text = ("We climbed Mount Everest before visiting Mount Fuji and Mount Etna.")
    predict(text)
    highlight_mountains(text)
  
