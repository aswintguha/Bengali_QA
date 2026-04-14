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

# Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_gsk_key_here
```

### 4. Run the App
```bash
.venv/bin/python -m uvicorn app:app --reload
```
Then open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

> **Note for macOS users:** Modern Homebrew Python blocks global `pip install`. Always use the `.venv` virtual environment as shown above.

---

## Daily Usage Guide

### ▶️ Starting the App
Every time you want to open the app, run these two commands in Terminal:
```bash
cd bengali_qa
.venv/bin/python -m uvicorn app:app --reload
```
Then open **http://127.0.0.1:8000** in your browser. The app is ready to use.

---

### ⏹️ Stopping the App
When you want to close the server and shut everything down:

1. Go to your **Terminal** where the server is running.
2. Press **`Ctrl + C`** — this immediately stops the server.
3. Close the Terminal window.

The browser tab will no longer load after this. Your generated CSV files are safely saved in the `output/` folder.

> **If you see `ERROR: Address already in use`** — a server is already running (possibly in another Terminal window). Run this to force-stop it:
> ```bash
> kill $(lsof -ti:8000)
> ```
> Then start the app again normally.

---

### 🔑 Changing Your Groq API Key
If you want to switch to a new API key:

1. Open the `.env` file in the `bengali_qa/` folder.
2. Replace the old key with your new one:
   ```
   GROQ_API_KEY=your_new_gsk_key_here
   ```
3. Save the file.
4. If the server is running, **stop it** (`Ctrl + C` in Terminal).
5. Start the server again:
   ```bash
   cd bengali_qa
   .venv/bin/python -m uvicorn app:app --reload
   ```
The server always reads the `.env` file fresh on startup, so the new key will be active immediately.

---

## Features

- **Double Source Support**: Process local PDF files or any Bengali news/article URL.
- **Smart PDF Extraction**: Uses **PyMuPDF** for reliable text extraction from digital PDFs.
- **OCR Fallback**: Supports **Tesseract OCR** for scanned/image-based Bengali PDFs.
- **High-Speed Generation**: Powered by Groq's Llama 3.3 70B (approx. 10x faster than Gemini).
- **Batch Processing**: Automatically splits large documents into chunks (~1 page each).
- **CSV Export**: Downloads a clean dataset with `Instruction`, `Response`, and `Category` (Factual, Causal, Reasoning).

---

## Project Structure

- `app.py`: FastAPI server logic & API integrations.
- `utils.py`: PDF processing, URL extraction, and text batching.
- `templates/`: Modern, responsive dashboard UI.
- `output/`: Automatically saves generated CSV datasets locally.
- `.env`: (Private) Stores your Groq API Key — never commit this to GitHub.
- `requirements.txt`: All Python dependencies.

---

## Tips for Best Results

- **PDF Quality**: Digital PDFs extract much faster than scanned ones.
- **URL Support**: Works best with single articles from sites like Prothom Alo, Anandabazar, etc.
- **Scale**: Generates ~10 Q&A pairs per page for high data density.
- **Free Tier**: Groq allows 14,400 requests/day and 30 requests/minute — more than enough for large books.
