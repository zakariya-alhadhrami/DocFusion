#!/usr/bin/env python3
"""
DocFusion — Gradio Web UI
Rihal CodeStacker 2026
"""

import os
os.system("apt-get install -y tesseract-ocr")
os.system("pip install pytesseract -q")

import re
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms, models
from transformers import LayoutLMForTokenClassification, BertTokenizerFast
from huggingface_hub import hf_hub_download
import gradio as gr

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS   = ["O", "B-VENDOR", "I-VENDOR", "B-DATE", "I-DATE", "B-TOTAL", "I-TOTAL"]
ID2LABEL = {i: x for i, x in enumerate(LABELS)}
LABEL2ID = {x: i for i, x in enumerate(LABELS)}

print("Loading models...")
tokenizer        = BertTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
extraction_model = LayoutLMForTokenClassification.from_pretrained(
    "Zakariya007/docfusion-v1",
    num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
)
extraction_model = extraction_model.to(DEVICE)
extraction_model.eval()

forgery_model = models.efficientnet_b0(weights=None)
forgery_model.classifier[1] = torch.nn.Linear(1280, 2)
weights_path = hf_hub_download(
    repo_id="Zakariya007/docfusion-v2",
    filename="efficientnet_best.pth"
)
forgery_model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
forgery_model = forgery_model.to(DEVICE)
forgery_model.eval()
print("✅ Models loaded!")

def extract_fields(image):
    try:
        import pytesseract
        ocr_text = pytesseract.image_to_string(image)

        # Regex-based extraction
        date_match = re.search(
            r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',
            ocr_text
        )
        date = date_match.group(0) if date_match else None

        total_match = re.search(
            r'(?:TOTAL|AMOUNT|JUMLAH)[^\d]*(\d+[\.,]\d{2})',
            ocr_text, re.IGNORECASE
        )
        total = total_match.group(1) if total_match else None

        lines  = [l.strip() for l in ocr_text.split('\n') if len(l.strip()) > 3]
        vendor = lines[0] if lines else None

        return vendor, date, total

    except Exception as e:
        print(f"Extraction error: {e}")
        return None, None, None


def detect_forgery(image, vendor, date, total):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output      = forgery_model(tensor)
        probs       = torch.softmax(output, dim=1)
        forged_prob = probs[0][1].item()
        visual_flag = forged_prob > 0.5

    rule_flags = []
    if not vendor: rule_flags.append("Missing vendor")
    if not date:   rule_flags.append("Missing date")
    if not total:  rule_flags.append("Missing total")

    if total:
        try:
            total_val = float(re.sub(r"[^\d.]", "", total))
            if total_val > 10000: rule_flags.append("Abnormally high total")
            if total_val <= 0:    rule_flags.append("Invalid total amount")
        except Exception:
            rule_flags.append("Unparseable total")

    if date:
        date_pattern = re.compile(
            r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2}"
        )
        if not date_pattern.search(date):
            rule_flags.append("Invalid date format")

    rule_flag = len(rule_flags) >= 2
    is_forged = 1 if (visual_flag or rule_flag) else 0
    return is_forged, forged_prob, rule_flags


def annotate_image(image, is_forged):
    annotated = image.copy()
    draw      = ImageDraw.Draw(annotated)
    w, h      = image.size
    if is_forged:
        draw.rectangle([0, 0, w-1, h-1], outline="#FF0000", width=6)
        draw.text((10, 10), "FORGED", fill="#FF0000")
    else:
        draw.rectangle([0, 0, w-1, h-1], outline="#00AA00", width=6)
        draw.text((10, 10), "GENUINE", fill="#00AA00")
    return annotated


def process_receipt(image):
    if image is None:
        return None, "Please upload an image.", "", "", "", ""

    pil_image = Image.fromarray(image).convert("RGB")
    vendor, date, total        = extract_fields(pil_image)
    is_forged, forged_prob, rule_flags = detect_forgery(pil_image, vendor, date, total)
    annotated                  = annotate_image(pil_image, is_forged)

    if is_forged:
        status = f"FORGED (confidence: {forged_prob:.1%})"
        if rule_flags:
            status += "\nFlags: " + ", ".join(rule_flags)
    else:
        status = f"GENUINE (forged probability: {forged_prob:.1%})"

    return (
        np.array(annotated),
        status,
        vendor or "Not detected",
        date   or "Not detected",
        total  or "Not detected",
        json.dumps({
            "vendor":    vendor,
            "date":      date,
            "total":     total,
            "is_forged": is_forged
        }, indent=2),
    )


with gr.Blocks(title="DocFusion") as demo:
    gr.Markdown("# DocFusion — Intelligent Document Processing")
    gr.Markdown("Upload a scanned receipt to extract fields and detect forgery.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Receipt", type="numpy")
            submit_btn  = gr.Button("Analyze Receipt", variant="primary")
        with gr.Column():
            output_image   = gr.Image(label="Annotated Receipt")
            forgery_status = gr.Textbox(label="Forgery Status", lines=4)

    with gr.Row():
        vendor_out = gr.Textbox(label="Vendor")
        date_out   = gr.Textbox(label="Date")
        total_out  = gr.Textbox(label="Total")

    json_out = gr.Code(label="JSON Output", language="json")

    submit_btn.click(
        fn      = process_receipt,
        inputs  = [input_image],
        outputs = [output_image, forgery_status, vendor_out, date_out, total_out, json_out],
    )

    gr.Markdown("**Models:** LayoutLM v1 (extraction) + EfficientNet-B0 (forgery)")

if __name__ == "__main__":
    demo.launch(share=True)
