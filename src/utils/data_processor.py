"""
Data processing utilities for issue classification.
"""

import pandas as pd
from models.classifier import classify_bug, classify_feature

def process_csv(file, bug_tokenizer, bug_model, feature_tokenizer, feature_model, hyperparams):
    """
    Process a CSV file containing issues and classify them.
    
    Args:
        file: CSV file object
        bug_tokenizer: Tokenizer for bug classification
        bug_model: Model for bug classification
        feature_tokenizer: Tokenizer for feature classification
        feature_model: Model for feature classification
        hyperparams: Hyperparameters for model inference
    
    Returns:
        tuple: (processed DataFrame, classification counts, bug issues, feature issues, improvement issues)
    """
    df = pd.read_csv(file)
    print("DEBUG: Length of DataFrame:", len(df))

    # Ensure 'title' and 'body' columns exist
    if "title" not in df.columns or "body" not in df.columns:
        raise ValueError("CSV must have 'title' and 'body' columns")

    # Convert to string and fill missing (NaN) with empty string
    df["title"] = df["title"].astype(str).fillna("")
    df["body"] = df["body"].astype(str).fillna("")

    # Combine into a single 'summary'
    df["summary"] = df["title"] + " " + df["body"]

    # Classify: Bug vs Non-Bug
    df["bug_classification"] = df["summary"].apply(
        lambda x: classify_bug(x, bug_tokenizer, bug_model, hyperparams)
    )

    # For Non-Bug, classify as Feature or Improvement
    non_bug_mask = df["bug_classification"] == "Non-Bug"
    df["feature_classification"] = ""
    df.loc[non_bug_mask, "feature_classification"] = df.loc[non_bug_mask, "summary"].apply(
        lambda x: classify_feature(x, feature_tokenizer, feature_model, hyperparams)
    )

    # Get classified issues
    bug_issues = df[df["bug_classification"] == "Bug"]
    feature_issues = df[(df["bug_classification"] == "Non-Bug") & (df["feature_classification"] == "Feature")]
    improvement_issues = df[(df["bug_classification"] == "Non-Bug") & (df["feature_classification"] == "Improvement")]

    classification_counts = {
        "bugs": len(bug_issues),
        "features": len(feature_issues),
        "improvements": len(improvement_issues)
    }

    return df, classification_counts, bug_issues, feature_issues, improvement_issues 