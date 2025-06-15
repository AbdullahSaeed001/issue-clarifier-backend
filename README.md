# Issue Clarifier Backend

A Flask-based backend service that classifies GitHub issues into categories (Bug, Feature, or Improvement) using BERT-based models.

## Project Structure

```
issue-clarifier-backend/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # API endpoints
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py          # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py      # Model loading and inference
│   │   ├── dataset.py         # Dataset handling
│   │   └── trainer.py         # Model training and evaluation
│   ├── utils/
│   │   ├── __init__.py
│   │   └── report_generator.py # PDF report generation
│   ├── __init__.py
│   └── app.py                 # Main application entry point
├── requirements.txt
└── README.md
```

## Features

- Issue classification using BERT models
- Support for both single text and CSV batch processing
- PDF report generation
- Model training and evaluation endpoints
- CORS support for frontend integration
- Modular code structure for better maintainability

## Setup

1. Clone the repository:

```bash
git clone https://github.com/AbdullahSaeed001/issue-clarifier-backend.git
cd issue-clarifier-backend
```

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the server:

```bash
python -m src.app
```

The server will start on `http://localhost:5000`

## API Endpoints

### Classify Issues

- **POST** `/classify`
  - Accepts either a text string or CSV file
  - Returns a PDF report with classification results
  - Headers include classification counts

### Train Models

- **POST** `/train`
  - Accepts a CSV file with training data
  - Trains both bug and feature classification models
  - Returns training results and hyperparameters

### Evaluate Models

- **POST** `/evaluate`
  - Accepts a CSV file with evaluation data
  - Optional query parameter `model` to specify which model to evaluate
  - Returns evaluation metrics

## Development

The codebase is organized into modules:

- `api`: Contains all API endpoints and request handling
- `config`: Configuration settings and constants
- `models`: Model-related code including classification, training, and evaluation
- `utils`: Utility functions for report generation and other tasks

## License

MIT
