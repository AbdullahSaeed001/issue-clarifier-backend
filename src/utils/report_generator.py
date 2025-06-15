"""
PDF report generation utilities.
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

def draw_text_with_wrap(pdf, text, x, y, width, font="Helvetica", size=12):
    """Draw text with word wrapping."""
    pdf.setFont(font, size)
    words = text.split()
    line = ""
    for word in words:
        test_line = line + " " + word if line else word
        if pdf.stringWidth(test_line, font, size) < width:
            line = test_line
        else:
            pdf.drawString(x, y, line)
            y -= 20
            line = word
    if line:
        pdf.drawString(x, y, line)
    return y

def generate_text_report(text, bug_class, feature_class=None):
    """Generate a PDF report for single text classification."""
    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter
    y_position = height - 50
    
    # Title
    pdf.setFont("Helvetica-Bold", 16)
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
    
    y_position = draw_text_with_wrap(pdf, text, 30, y_position, width - 60)
    
    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer

def generate_csv_report(df, classification_counts, bug_issues, feature_issues, improvement_issues, inference_time):
    """Generate a PDF report for CSV classification results."""
    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter
    y_position = height - 50
    
    # Title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(150, y_position, "Enhanced Issue Classification Report")
    y_position -= 40
    
    # Summary
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
    
    # List issues by category
    for category, issues in [
        ("Bugs", bug_issues),
        ("Features", feature_issues),
        ("Improvements", improvement_issues)
    ]:
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(30, y_position, category)
        y_position -= 20
        pdf.setFont("Helvetica", 12)
        
        for idx, row in issues.iterrows():
            line = f"- {str(row['title'])[:80]}"
            pdf.drawString(30, y_position, line)
            y_position -= 20
            if y_position < 40:
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                y_position = height - 40
    
    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer 