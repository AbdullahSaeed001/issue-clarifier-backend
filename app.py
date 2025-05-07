from flask import Flask, request, send_file, jsonify, make_response
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import json
import time
from flask_cors import CORS
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

app = Flask(__name__)
CORS(app, expose_headers=["X-Classification-Counts"])

########################################
# Configuration and Hyperparameters
########################################

def load_hyperparameters():
    return {
        "max_length": 256,
        "learning_rate": 5e-5,
        "batch_size": 8,
        "epochs": 2  # For demo/testing
    }

hyperparams = load_hyperparameters()

########################################
# Global Models and Tokenizers
########################################

BUG_MODEL_NAME = "prajjwal1/bert-small"
FEATURE_MODEL_NAME = "prajjwal1/bert-small"

bug_tokenizer = AutoTokenizer.from_pretrained(BUG_MODEL_NAME)
bug_model = AutoModelForSequenceClassification.from_pretrained(BUG_MODEL_NAME, num_labels=2)
bug_model.eval()

feature_tokenizer = AutoTokenizer.from_pretrained(FEATURE_MODEL_NAME)
feature_model = AutoModelForSequenceClassification.from_pretrained(FEATURE_MODEL_NAME, num_labels=2)
feature_model.eval()

BUG_LABELS = ["Bug", "Non-Bug"]
FEATURE_LABELS = ["Feature", "Improvement"]

########################################
# Inference Functions
########################################

def classify_bug(text):
    encoded = bug_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=hyperparams["max_length"],
        return_tensors="pt"
    )
    with torch.no_grad():
        output = bug_model(**encoded)
    pred = torch.argmax(output.logits, dim=1).item()
    return BUG_LABELS[pred]

def classify_feature(text):
    encoded = feature_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=hyperparams["max_length"],
        return_tensors="pt"
    )
    with torch.no_grad():
        output = feature_model(**encoded)
    pred = torch.argmax(output.logits, dim=1).item()
    return FEATURE_LABELS[pred]

########################################
# Dataset for Training/Evaluation
########################################

class IssueDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

########################################
# CSV Processing for Inference
########################################

def process_csv(file):
    df = pd.read_csv(file)
    print("DEBUG: Length of DataFrame:", len(df))

    # Ensure 'title' and 'body' columns exist
    if "title" not in df.columns or "body" not in df.columns:
        raise ValueError("CSV must have 'title' and 'body' columns")

    # Convert to string and fill missing (NaN) with empty string
    df["title"] = df["title"].astype(str).fillna("")
    df["body"]  = df["body"].astype(str).fillna("")

    # Combine into a single 'summary'
    df["summary"] = df["title"] + " " + df["body"]

    # Classify: Bug vs Non-Bug
    df["bug_classification"] = df["summary"].apply(classify_bug)

    # For Non-Bug, classify as Feature or Improvement
    non_bug_mask = df["bug_classification"] == "Non-Bug"
    df["feature_classification"] = ""
    df.loc[non_bug_mask, "feature_classification"] = df.loc[non_bug_mask, "summary"].apply(classify_feature)

    bug_issues = df[df["bug_classification"] == "Bug"]
    feature_issues = df[(df["bug_classification"] == "Non-Bug") & (df["feature_classification"] == "Feature")]
    improvement_issues = df[(df["bug_classification"] == "Non-Bug") & (df["feature_classification"] == "Improvement")]

    classification_counts = {
        "bugs": len(bug_issues),
        "features": len(feature_issues),
        "improvements": len(improvement_issues)
    }

    return df, classification_counts, bug_issues, feature_issues, improvement_issues

########################################
# Training Functions
########################################

def train_model(model, tokenizer, df, label_mapping, hyperparams):
    if not {"title", "body", "label"}.issubset(df.columns):
        raise ValueError("CSV must have 'title', 'body', and 'label' columns")

    df["title"] = df["title"].astype(str).fillna("")
    df["body"]  = df["body"].astype(str).fillna("")
    df["summary"] = df["title"] + " " + df["body"]

    df["label_int"] = df["label"].map(label_mapping)
    texts = df["summary"].tolist()
    labels = df["label_int"].tolist()
    
    dataset = IssueDataset(texts, labels, tokenizer, hyperparams["max_length"])
    dataloader = DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    
    for epoch in range(hyperparams["epochs"]):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{hyperparams['epochs']} - Loss: {avg_loss:.4f}")
    
    model.eval()
    return model

def train_bug_model(df, hyperparams):
    label_mapping = {"Bug": 0, "Non-Bug": 1}
    print("🔹 Training bug model...")
    trained = train_model(bug_model, bug_tokenizer, df, label_mapping, hyperparams)
    return trained

def train_feature_model(df, hyperparams):
    label_mapping = {"Feature": 0, "Improvement": 1}
    print("🔹 Training feature model...")
    trained = train_model(feature_model, feature_tokenizer, df, label_mapping, hyperparams)
    return trained

########################################
# Evaluation Functions
########################################

def evaluate_model(model, tokenizer, df, label_mapping, hyperparams):
    if not {"title", "body", "label"}.issubset(df.columns):
        raise ValueError("CSV must have 'title', 'body', and 'label' columns")

    df["title"] = df["title"].astype(str).fillna("")
    df["body"]  = df["body"].astype(str).fillna("")
    df["summary"] = df["title"] + " " + df["body"]

    df["label_int"] = df["label"].map(label_mapping)
    texts = df["summary"].tolist()
    labels = df["label_int"].tolist()
    
    dataset = IssueDataset(texts, labels, tokenizer, hyperparams["max_length"])
    dataloader = DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    total, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            labels_batch = batch["labels"].to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy}

def evaluate_bug_model(df, hyperparams):
    label_mapping = {"Bug": 0, "Non-Bug": 1}
    print("🔹 Evaluating bug model...")
    result = evaluate_model(bug_model, bug_tokenizer, df, label_mapping, hyperparams)
    print("Bug model evaluation result:", result)
    return result

def evaluate_feature_model(df, hyperparams):
    label_mapping = {"Feature": 0, "Improvement": 1}
    print("🔹 Evaluating feature model...")
    result = evaluate_model(feature_model, feature_tokenizer, df, label_mapping, hyperparams)
    print("Feature model evaluation result:", result)
    return result

########################################
# Flask Endpoints
########################################

@app.route("/classify", methods=["POST"])
def classify_csv():
    try:
        print("🔹 Received request at /classify")
        
        # Handle text-based classification
        if "text" in request.form:
            text = request.form["text"]
            print("🔹 Processing text classification")
            
            # Classify the text
            bug_class = classify_bug(text)
            feature_class = classify_feature(text) if bug_class == "Non-Bug" else None
            
            # Create classification counts
            classification_counts = {
                "bugs": 1 if bug_class == "Bug" else 0,
                "features": 1 if feature_class == "Feature" else 0,
                "improvements": 1 if feature_class == "Improvement" else 0
            }
            
            # Generate PDF report
            pdf_buffer = io.BytesIO()
            pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
            width, height = letter
            
            # Title
            pdf.setFont("Helvetica-Bold", 16)
            y_position = height - 50
            pdf.drawString(150, y_position, "Issue Classification Report")
            y_position -= 40
            
            # Classification details
            pdf.setFont("Helvetica", 12)
            pdf.drawString(30, y_position, "Text Analysis Results:")
            y_position -= 30
            
            pdf.drawString(30, y_position, f"Primary Classification: {bug_class}")
            y_position -= 20
            
            if bug_class == "Non-Bug":
                pdf.drawString(30, y_position, f"Secondary Classification: {feature_class}")
                y_position -= 20
            
            # Original text
            y_position -= 20
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(30, y_position, "Original Text:")
            y_position -= 20
            pdf.setFont("Helvetica", 12)
            
            # Word wrap the text
            words = text.split()
            line = ""
            for word in words:
                test_line = line + " " + word if line else word
                if pdf.stringWidth(test_line, "Helvetica", 12) < width - 60:
                    line = test_line
                else:
                    pdf.drawString(30, y_position, line)
                    y_position -= 20
                    line = word
            if line:
                pdf.drawString(30, y_position, line)
            
            pdf.save()
            pdf_buffer.seek(0)
            
            response = make_response(
                send_file(
                    pdf_buffer,
                    mimetype="application/pdf",
                    as_attachment=True,
                    download_name="text_classification_report.pdf"
                )
            )
            response.headers["X-Classification-Counts"] = json.dumps(classification_counts)
            response.headers["Access-Control-Expose-Headers"] = "X-Classification-Counts"
            return response
            
        # Handle CSV file classification (existing code)
        if "csvFile" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["csvFile"]
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400
        
        # Process the CSV (with missing-value fix)
        start_time = time.time()
        df, classification_counts, bug_issues, feature_issues, improvement_issues = process_csv(file)
        inference_time = time.time() - start_time
        
        pdf_buffer = io.BytesIO()
        pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        pdf.setFont("Helvetica-Bold", 16)
        y_position = height - 50
        pdf.drawString(150, y_position, "Enhanced Issue Classification Report")
        y_position -= 40
        
        pdf.setFont("Helvetica", 12)
        pdf.drawString(30, y_position, f"Total Issues: {len(df)}")
        y_position -= 20
        pdf.drawString(30, y_position, f"Bugs: {classification_counts['bugs']}")
        y_position -= 20
        pdf.drawString(30, y_position, f"Features: {classification_counts['features']}")
        y_position -= 20
        pdf.drawString(30, y_position, f"Improvements: {classification_counts['improvements']}")
        y_position -= 20
        pdf.drawString(30, y_position, f"Inference Time: {inference_time:.2f} seconds")
        y_position -= 30
        
        # List bug issues
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(30, y_position, "Bugs")
        y_position -= 20
        pdf.setFont("Helvetica", 12)
        for idx, row in bug_issues.iterrows():
            # Safely convert to string and slice
            line = f"- {str(row['title'])[:80]}"
            pdf.drawString(30, y_position, line)
            y_position -= 20
            if y_position < 40:
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                y_position = height - 40
        
        # List feature issues
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(30, y_position, "Features")
        y_position -= 20
        pdf.setFont("Helvetica", 12)
        for idx, row in feature_issues.iterrows():
            line = f"- {str(row['title'])[:80]}"
            pdf.drawString(30, y_position, line)
            y_position -= 20
            if y_position < 40:
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                y_position = height - 40
        
        # List improvement issues
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(30, y_position, "Improvements")
        y_position -= 20
        pdf.setFont("Helvetica", 12)
        for idx, row in improvement_issues.iterrows():
            line = f"- {str(row['title'])[:80]}"
            pdf.drawString(30, y_position, line)
            y_position -= 20
            if y_position < 40:
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                y_position = height - 40
        
        pdf.save()
        pdf_buffer.seek(0)
        
        response = make_response(
            send_file(
                pdf_buffer,
                mimetype="application/pdf",
                as_attachment=True,
                download_name="enhanced_classification_report.pdf"
            )
        )
        response.headers["X-Classification-Counts"] = json.dumps(classification_counts)
        response.headers["Access-Control-Expose-Headers"] = "X-Classification-Counts"
        return response
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train():
    try:
        print("🔹 Received training request at /train")
        if "csvFile" not in request.files:
            return jsonify({"error": "No training CSV file provided"}), 400
        file = request.files["csvFile"]
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400
        
        df = pd.read_csv(file)
        global bug_model, feature_model
        results = {}
        bug_model = train_bug_model(df, hyperparams)
        bug_model.save_pretrained("bug_checkpoint")
        results["bug_model"] = "Trained and updated."
        feature_model = train_feature_model(df, hyperparams)
        feature_model.save_pretrained("feature_checkpoint")
        results["feature_model"] = "Trained and updated."
        
        print("🔹 Training complete. Details:", results)
        print("🔹 Current Hyperparameters:", hyperparams)
        return jsonify({"message": "Training complete.", "details": results, "hyperparams": hyperparams})
    except Exception as e:
        print(f"❌ Training Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        print("🔹 Received evaluation request at /evaluate")
        if "csvFile" not in request.files:
            return jsonify({"error": "No evaluation CSV file provided"}), 400
        file = request.files["csvFile"]
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400
        
        model_to_eval = request.args.get("model", "both")
        df = pd.read_csv(file)
        results = {}
        if model_to_eval in ["bug", "both"]:
            res_bug = evaluate_bug_model(df, hyperparams)
            results["bug_model"] = res_bug
        if model_to_eval in ["feature", "both"]:
            res_feature = evaluate_feature_model(df, hyperparams)
            results["feature_model"] = res_feature
        
        print("🔹 Evaluation complete. Results:", results)
        return jsonify({"message": "Evaluation complete.", "results": results})
    except Exception as e:
        print(f"❌ Evaluation Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
