#!/usr/bin/env python3
"""
DocFusion Solution — Rihal CodeStacker 2026
Author: Zakariya007
Models:
    docfusion-v1 → LayoutLM v1 (field extraction)
    docfusion-v2 → EfficientNet-B0 (forgery detection)
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path


class DocFusionSolution:
    """End-to-end document processing pipeline for scanned receipts."""

    HF_EXTRACTION_MODEL = "Zakariya007/docfusion-v1"
    HF_FORGERY_MODEL    = "Zakariya007/docfusion-v2"
    FORGERY_WEIGHTS     = "efficientnet_best.pth"
    MAX_LENGTH          = 512
    IMG_SIZE            = 224
    FORGERY_THRESHOLD   = 0.5

    # ------------------------------------------------------------------ #
    #  TRAIN                                                               #
    # ------------------------------------------------------------------ #
    def train(self, train_dir: str, work_dir: str) -> str:
        """
        Download pretrained models from HuggingFace Hub into work_dir.
        Returns the model directory path.
        """
        import torch
        from transformers import (
            LayoutLMForTokenClassification,
            BertTokenizerFast,
        )
        from torchvision import models
        from huggingface_hub import hf_hub_download

        model_dir = os.path.join(work_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        print("[train] Downloading extraction model (docfusion-v1)...")
        tokenizer = BertTokenizerFast.from_pretrained(
            "microsoft/layoutlm-base-uncased"
        )
        extraction_model = LayoutLMForTokenClassification.from_pretrained(
            self.HF_EXTRACTION_MODEL
        )
        tokenizer.save_pretrained(os.path.join(model_dir, "extraction"))
        extraction_model.save_pretrained(os.path.join(model_dir, "extraction"))
        print("[train] Extraction model saved.")

        print("[train] Downloading forgery model (docfusion-v2)...")
        weights_path = hf_hub_download(
            repo_id=self.HF_FORGERY_MODEL,
            filename=self.FORGERY_WEIGHTS,
        )
        shutil.copy(weights_path, os.path.join(model_dir, self.FORGERY_WEIGHTS))
        print("[train] Forgery model saved.")

        return model_dir

    # ------------------------------------------------------------------ #
    #  PREDICT                                                             #
    # ------------------------------------------------------------------ #
    def predict(
        self,
        model_dir:        str,
        test_dir:         str,
        predictions_path: str,
    ) -> None:
        """
        Run inference on all records in test.jsonl and write predictions.jsonl.
        """
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[predict] Using device: {device}")

        # Load models
        extraction_model, tokenizer = self._load_extraction_model(model_dir, device)
        forgery_model               = self._load_forgery_model(model_dir, device)

        # Load test records
        test_jsonl = os.path.join(test_dir, "test.jsonl")
        records    = self._load_jsonl(test_jsonl)
        print(f"[predict] Running inference on {len(records)} records...")

        predictions = []
        for record in records:
            pred = self._predict_single(
                record          = record,
                test_dir        = test_dir,
                extraction_model= extraction_model,
                tokenizer       = tokenizer,
                forgery_model   = forgery_model,
                device          = device,
            )
            predictions.append(pred)
            print(f"[predict] {record['id']} → forged={pred['is_forged']} "
                  f"vendor={pred.get('vendor')} date={pred.get('date')} total={pred.get('total')}")

        # Write predictions.jsonl
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        with open(predictions_path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

        print(f"[predict] Wrote {len(predictions)} predictions to {predictions_path}")

    # ------------------------------------------------------------------ #
    #  INTERNAL: Load Models                                               #
    # ------------------------------------------------------------------ #
    def _load_extraction_model(self, model_dir: str, device):
        from transformers import LayoutLMForTokenClassification, BertTokenizerFast
        import torch

        labels   = ["O", "B-VENDOR", "I-VENDOR", "B-DATE", "I-DATE", "B-TOTAL", "I-TOTAL"]
        label2id = {x: i for i, x in enumerate(labels)}
        id2label = {i: x for i, x in enumerate(labels)}

        tokenizer = BertTokenizerFast.from_pretrained(
            os.path.join(model_dir, "extraction")
        )
        model = LayoutLMForTokenClassification.from_pretrained(
            os.path.join(model_dir, "extraction"),
            num_labels = len(labels),
            id2label   = id2label,
            label2id   = label2id,
        )
        model = model.to(device)
        model.eval()
        return model, tokenizer

    def _load_forgery_model(self, model_dir: str, device):
        import torch
        from torchvision import models

        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = __import__("torch.nn", fromlist=["Linear"]).Linear(1280, 2)

        weights_path = os.path.join(model_dir, self.FORGERY_WEIGHTS)
        state_dict   = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    #  INTERNAL: Single Prediction                                         #
    # ------------------------------------------------------------------ #
    def _predict_single(
        self,
        record,
        test_dir,
        extraction_model,
        tokenizer,
        forgery_model,
        device,
    ) -> dict:
        import torch
        from PIL import Image

        record_id  = record["id"]
        image_path = os.path.join(test_dir, record["image_path"])

        # ── Field Extraction ──────────────────────────────────────────
        # Use fields from test.jsonl if available (already extracted)
        fields = record.get("fields", {})
        vendor = fields.get("vendor") or None
        date   = fields.get("date")   or None
        total  = fields.get("total")  or None

        # If fields are missing run LayoutLM extraction
        if not any([vendor, date, total]):
            vendor, date, total = self._extract_fields(
                image_path       = image_path,
                extraction_model = extraction_model,
                tokenizer        = tokenizer,
                device           = device,
            )

        # ── Forgery Detection ─────────────────────────────────────────
        is_forged = self._detect_forgery(
            image_path    = image_path,
            forgery_model = forgery_model,
            device        = device,
            vendor        = vendor,
            date          = date,
            total         = total,
        )

        return {
            "id":        record_id,
            "is_forged": is_forged,
            "vendor":    vendor,
            "date":      date,
            "total":     total,
        }

    # ------------------------------------------------------------------ #
    #  INTERNAL: Field Extraction                                          #
    # ------------------------------------------------------------------ #
    def _extract_fields(self, image_path, extraction_model, tokenizer, device):
        """Run LayoutLM v1 to extract vendor, date, total from receipt."""
        import torch
        from PIL import Image

        id2label = {
            0: "O", 1: "B-VENDOR", 2: "I-VENDOR",
            3: "B-DATE", 4: "I-DATE",
            5: "B-TOTAL", 6: "I-TOTAL"
        }

        try:
            # Use pytesseract for OCR if available
            import pytesseract
            image = Image.open(image_path).convert("RGB")
            ocr_data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT
            )

            words, boxes = [], []
            w, h = image.size
            for i, word in enumerate(ocr_data["text"]):
                if word.strip():
                    x, y      = ocr_data["left"][i], ocr_data["top"][i]
                    bw, bh    = ocr_data["width"][i], ocr_data["height"][i]
                    x1 = int(x / w * 1000)
                    y1 = int(y / h * 1000)
                    x2 = int((x + bw) / w * 1000)
                    y2 = int((y + bh) / h * 1000)
                    words.append(word.strip())
                    boxes.append([
                        min(x1, 1000), min(y1, 1000),
                        min(x2, 1000), min(y2, 1000)
                    ])

            if not words:
                return None, None, None

            encoding = tokenizer(
                words,
                boxes           = boxes,
                is_split_into_words = True,
                padding         = "max_length",
                truncation      = True,
                max_length      = self.MAX_LENGTH,
                return_tensors  = "pt",
            )

            # Align boxes to tokens
            word_ids    = encoding.word_ids(batch_index=0)
            token_boxes = []
            for word_idx in word_ids:
                if word_idx is None:
                    token_boxes.append([0, 0, 0, 0])
                else:
                    token_boxes.append(boxes[word_idx])

            encoding["bbox"] = torch.tensor([token_boxes], dtype=torch.long)
            inputs = {k: v.to(device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs    = extraction_model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)[0]

            # Decode predictions
            vendor_tokens, date_tokens, total_tokens = [], [], []
            for token_idx, (word_idx, pred_id) in enumerate(
                zip(word_ids, predictions.tolist())
            ):
                if word_idx is None:
                    continue
                label = id2label.get(pred_id, "O")
                token = tokenizer.convert_ids_to_tokens(
                    encoding["input_ids"][0][token_idx].item()
                )
                token = token.replace("##", "")
                if label in ("B-VENDOR", "I-VENDOR"):
                    vendor_tokens.append(token)
                elif label in ("B-DATE", "I-DATE"):
                    date_tokens.append(token)
                elif label in ("B-TOTAL", "I-TOTAL"):
                    total_tokens.append(token)

            vendor = " ".join(vendor_tokens).strip() or None
            date   = " ".join(date_tokens).strip()   or None
            total  = " ".join(total_tokens).strip()  or None
            return vendor, date, total

        except Exception as e:
            print(f"[extract] Error: {e}")
            return None, None, None

    # ------------------------------------------------------------------ #
    #  INTERNAL: Forgery Detection (Hybrid)                                #
    # ------------------------------------------------------------------ #
    def _detect_forgery(
        self,
        image_path,
        forgery_model,
        device,
        vendor=None,
        date=None,
        total=None,
    ) -> int:
        """
        Hybrid forgery detection:
        1. EfficientNet visual score
        2. Rule-based anomaly checks
        """
        import torch
        from torchvision import transforms
        from PIL import Image

        visual_forged = 0

        # ── Visual Check (EfficientNet) ───────────────────────────────
        try:
            transform = transforms.Compose([
                transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                ),
            ])
            image  = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output       = forgery_model(tensor)
                probs        = torch.softmax(output, dim=1)
                forged_prob  = probs[0][1].item()
                visual_forged = 1 if forged_prob > self.FORGERY_THRESHOLD else 0

        except Exception as e:
            print(f"[forgery] Visual check error: {e}")

        # ── Rule-Based Checks ─────────────────────────────────────────
        rule_flags = []

        # Check 1 — Missing critical fields
        if not vendor:
            rule_flags.append("missing_vendor")
        if not date:
            rule_flags.append("missing_date")
        if not total:
            rule_flags.append("missing_total")

        # Check 2 — Abnormally high total
        if total:
            try:
                total_val = float(re.sub(r"[^\d.]", "", total))
                if total_val > 10000:
                    rule_flags.append("abnormally_high_total")
                if total_val <= 0:
                    rule_flags.append("invalid_total")
            except Exception:
                rule_flags.append("unparseable_total")

        # Check 3 — Invalid date format
        if date:
            date_pattern = re.compile(
                r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2}"
            )
            if not date_pattern.search(date):
                rule_flags.append("invalid_date_format")

        rule_forged = 1 if len(rule_flags) >= 2 else 0

        # ── Combine: visual OR rule-based ─────────────────────────────
        is_forged = 1 if (visual_forged == 1 or rule_forged == 1) else 0
        return is_forged

    # ------------------------------------------------------------------ #
    #  INTERNAL: Helpers                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_jsonl(path: str) -> list[dict]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
