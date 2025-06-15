"""
Configuration settings for the Issue Clarifier application.
"""

# Model Names
BUG_MODEL_NAME = "AbdullahSaeed001/CategorizeIT-Bug"
FEATURE_MODEL_NAME = "AbdullahSaeed001/CategorizeIT-Feature"

# Labels
BUG_LABELS = ["Bug", "Non-Bug"]
FEATURE_LABELS = ["Feature", "Improvement"]

def load_hyperparameters():
    """Load hyperparameters for model training and inference."""
    return {
        "max_length": 256,
        "learning_rate": 5e-5,
        "batch_size": 8,
        "epochs": 2  # For demo/testing
    }

# Flask Configuration
FLASK_CONFIG = {
    "debug": True,
    "port": 5000
} 