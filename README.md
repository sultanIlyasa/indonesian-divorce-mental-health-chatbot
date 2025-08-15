# 🧠 Indonesian Divorce Mental Health Chatbot
**Transformer from Scratch – TensorFlow + Gradio**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)  
![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-green.svg)  
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)  

A web-based chatbot application built entirely from scratch using the Transformer architecture (Encoder-Decoder) in TensorFlow.  
The chatbot provides empathetic support for **children affected by parental divorce**, trained on a custom Indonesian dataset based on **John Bowlby’s Positive Attachment Theory**.

---

## 📌 Features
- ✅ Transformer Encoder-Decoder built **from scratch** (no pre-trained models)
- ✅ Custom **Indonesian tokenizer** with `SubwordTextEncoder`
- ✅ Empathetic & context-aware responses based on **attachment theory**
- ✅ **Multi-tab UI**: Chat interface + Development info page
- ✅ Custom avatars, privacy-friendly design
- ✅ Evaluation with **BLEU & METEOR** metrics

---

## 🛠 Tech Stack
- **Language:** Python 3.9+
- **Frameworks/Libraries:** TensorFlow 2.x, TensorFlow Datasets, Gradio, NumPy, Matplotlib
- **Interface:** Gradio Blocks + ChatInterface
- **Model Type:** Transformer Encoder-Decoder

---

## 🏗 Architecture Overview
```text
Data → Tokenizer → Positional Encoding → Encoder Layers → Decoder Layers
→ Multi-Head Attention → Dense Output → Response Generation

```

---
## Installation
- git clone https://github.com/yourusername/mental-health-chatbot-transformer.git
- cd mental-health-chatbot-transformer
- pip install -r requirements.txt

---
## Usage
```text
python app.py
```

---
## 📂 Dataset
- Size: 525 conversational pairs
- Categories: Opening statements, parental context, termination
- Based on: John Bowlby’s Positive Attachment Theory
- Includes professional help triggers for high-risk cases

---
## ⚙ How It Works
- User input → tokenized & padded → START/END tokens added
- Transformer model generates output via autoregressive decoding
- Masking ensures no future token or padding leakage in attention
- Final output is detokenized back into natural language

---
## Evaluation
- BLEU Score – Token-level similarity with reference responses
- METEOR Score – Semantic similarity at word-level
- Qualitative Review – Checked for empathy & context accuracy
