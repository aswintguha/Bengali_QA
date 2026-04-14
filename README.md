# Bengali Q&A Dataset Generator

A powerful, high-speed tool to automatically generate Bengali Question & Answer datasets from PDF documents or website URLs. Powered by **Groq + Llama 3.3 70B** for lightning-fast, high-quality Bengali language processing.

---

## Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.9+ installed and get a **free API Key** from [Groq Console](https://console.groq.com/keys).

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/aswintguha/Bengali_QA.git
cd Bengali_QA

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```bash
echo "GROQ_API_KEY=your_gsk_key_here" > .env
```

### 4. Run the App
```bash
python3 -m uvicorn app:app --reload
```
Visit **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

##  Features

- **Double Source Support**: Process local PDF files or any Bengali news/article URL.
- **Smart PDF Extraction**: Uses **PyMuPDF** for reliable text extraction from digital PDFs.
- **OCR Fallback**: Supports **Tesseract OCR** for scanned/image-based Bengali PDFs.
- **High-Speed Generation**: Powered by Groq's Llama 3.3 70B (approx. 10x faster than Gemini).
- **Batch Processing**: Automatically splits large documents into chunks.
- **CSV Export**: Downloads a clean dataset with `Instruction`, `Response`, and `Category` (Factual, Causal, Reasoning).

---

## 🛠 Project Structure

- `app.py`: FastAPI server logic & API integrations.
- `utils.py`: PDF processing, URL extraction, and text batching.
- `templates/`: Modern, responsive dashboard UI.
- `output/`: Automatically saves generated CSV datasets locally.
- `.env`: (Private) Stores your Groq API Key.

---

## 💡 Tips for Best Results

- **PDF Quality**: Digital PDFs extract much faster than scanned ones.
- **URL Limit**: Works best with single articles from sites like Prothom Alo, Anandabazar, etc.
- **Scale**: The system now generates ~10 questions per page to ensure high data density.

---

## ⚖️ License
MIT License. Free to use and modify for dataset creation projects.
