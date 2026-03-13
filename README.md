

# 🧾 DocFusion — Intelligent Document Processing

**Rihal CodeStacker 2026 — ML Challenge**

An end-to-end pipeline for processing scanned receipts. Upload a receipt image to automatically extract key fields and detect potential forgeries.

---

## 🚀 Features

- **Field Extraction** — Automatically extracts vendor, date, and total from receipt images using LayoutLM v1
- **Forgery Detection** — Hybrid approach combining EfficientNet-B0 visual analysis with rule-based anomaly detection
- **Visual Annotation** — Highlights extracted fields with colored bounding boxes directly on the receipt image
- **JSON Output** — Returns structured JSON for downstream processing

---

## 🏗️ Architecture

```
Receipt Image
      ↓
┌─────────────────────────┐
│   Field Extraction      │
│   LayoutLM v1           │
│   Text + Layout         │
│   → vendor, date, total │
└─────────────────────────┘
      +
┌─────────────────────────┐
│   Forgery Detection     │
│   EfficientNet-B0       │
│   Visual Analysis       │
│   + Rule-based Checks   │
│   → is_forged (0/1)     │
└─────────────────────────┘
      ↓
  Final Output JSON
```

---

## 🤖 Models

| Model | HuggingFace | Purpose |
|---|---|---|
| LayoutLM v1 | [Zakariya007/docfusion-v1](https://huggingface.co/Zakariya007/docfusion-v1) | Field extraction |
| EfficientNet-B0 | [Zakariya007/docfusion-v2](https://huggingface.co/Zakariya007/docfusion-v2) | Forgery detection |

---

## 📊 Training Data

| Dataset | Samples | Purpose |
|---|---|---|
| SROIE 2019 | 626 train / 347 test | Field extraction |
| CORD | 800 train / 100 val / 100 test | Field extraction |
| Find-It-Again | 577 train / 193 val / 218 test | Forgery detection |

---

## 📈 Model Performance

| Task | Metric | Score |
|---|---|---|
| Field Extraction | F1 | 0.731 |
| Forgery Detection | Accuracy | ~80% |

---

## 🔍 Forgery Detection Logic

The hybrid forgery detection combines two signals:

**Visual (EfficientNet-B0):**
- Analyzes receipt image for pixel-level tampering
- Detects copy-paste and paint-based modifications

**Rule-based checks:**
- Missing critical fields (vendor, date, total)
- Abnormally high total amount (> 10,000)
- Invalid date format
- Unparseable total value

A receipt is flagged as forged if either the visual model or 2+ rule-based checks trigger.

---

## 🎨 Bounding Box Colors

| Color | Field |
|---|---|
| 🟢 Green | Vendor |
| 🔵 Blue | Date |
| 🟠 Orange | Total |
| 🔴 Red border | Forged receipt |

---

## 💻 Local Setup

```bash
git clone https://huggingface.co/spaces/Zakariya007/docfusion
cd docfusion
pip install -r requirements.txt
apt-get install tesseract-ocr
python app.py
```

---

## 📁 Project Structure

```
DocFusion/
├── app.py              ← Gradio UI
├── solution.py         ← Competition harness
├── requirements.txt    ← Dependencies
├── README.md           ← This file
└── notebooks/
    ├── eda.ipynb
    ├── training.ipynb
    └── forgery_training.ipynb
```

---

## 👤 Author

**Zakariya** — Rihal CodeStacker 2026 Participant

---

*Built with ❤️ using HuggingFace Transformers, PyTorch, and Gradio*
