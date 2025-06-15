"""
Main application entry point for the Issue Clarifier.
"""

import time
import json
import pandas as pd
from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS

from config.config import load_hyperparameters, FLASK_CONFIG
from models.classifier import load_models, classify_bug, classify_feature
from models.trainer import train_model, evaluate_model
from utils.report_generator import generate_text_report, generate_csv_report
from utils.data_processor import process_csv

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app, expose_headers=["X-Classification-Counts"])
    
    # Load hyperparameters
    hyperparams = load_hyperparameters()
    
    # Load models
    bug_tokenizer, bug_model, feature_tokenizer, feature_model = load_models(hyperparams)
    
    @app.route("/classify", methods=["POST"])
    def classify_csv():
        try:
            print("🔹 Received request at /classify")
            
            # Handle text-based classification
            if "text" in request.form:
                text = request.form["text"]
                print("🔹 Processing text classification")
                
                # Classify the text
                bug_class = classify_bug(text, bug_tokenizer, bug_model, hyperparams)
                feature_class = classify_feature(text, feature_tokenizer, feature_model, hyperparams) if bug_class == "Non-Bug" else None
                
                # Create classification counts
                classification_counts = {
                    "bugs": 1 if bug_class == "Bug" else 0,
                    "features": 1 if feature_class == "Feature" else 0,
                    "improvements": 1 if feature_class == "Improvement" else 0
                }
                
                # Generate PDF report
                pdf_buffer = generate_text_report(text, bug_class, feature_class)
                
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
            
            # Handle CSV file classification
            if "csvFile" not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            file = request.files["csvFile"]
            if not file.filename.endswith(".csv"):
                return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400
            
            # Process the CSV
            start_time = time.time()
            df, classification_counts, bug_issues, feature_issues, improvement_issues = process_csv(
                file, bug_tokenizer, bug_model, feature_tokenizer, feature_model, hyperparams
            )
            inference_time = time.time() - start_time
            
            # Generate PDF report
            pdf_buffer = generate_csv_report(
                df, classification_counts, bug_issues, feature_issues, improvement_issues, inference_time
            )
            
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
            results = {}
            
            # Train bug model
            bug_model = train_model(bug_model, bug_tokenizer, df, {"Bug": 0, "Non-Bug": 1}, hyperparams)
            bug_model.save_pretrained("bug_checkpoint")
            results["bug_model"] = "Trained and updated."
            
            # Train feature model
            feature_model = train_model(feature_model, feature_tokenizer, df, {"Feature": 0, "Improvement": 1}, hyperparams)
            feature_model.save_pretrained("feature_checkpoint")
            results["feature_model"] = "Trained and updated."
            
            print("🔹 Training complete. Details:", results)
            print("🔹 Current Hyperparameters:", hyperparams)
            return jsonify({
                "message": "Training complete.",
                "details": results,
                "hyperparams": hyperparams
            })
            
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
                res_bug = evaluate_model(bug_model, bug_tokenizer, df, {"Bug": 0, "Non-Bug": 1}, hyperparams)
                results["bug_model"] = res_bug
            
            if model_to_eval in ["feature", "both"]:
                res_feature = evaluate_model(feature_model, feature_tokenizer, df, {"Feature": 0, "Improvement": 1}, hyperparams)
                results["feature_model"] = res_feature
            
            print("🔹 Evaluation complete. Results:", results)
            return jsonify({
                "message": "Evaluation complete.",
                "results": results
            })
            
        except Exception as e:
            print(f"❌ Evaluation Error: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(
        debug=FLASK_CONFIG["debug"],
        port=FLASK_CONFIG["port"]
    )
    