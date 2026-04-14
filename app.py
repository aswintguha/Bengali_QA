import os
import io
import time
import csv
import re

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from groq import Groq
from utils import extract_text_from_pdf, extract_text_from_url, split_into_batches

# ── Load environment variables ──
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not found. Please add it to your .env file.")

# ── Groq + Llama 3.3 70B (free, fast, generous limits) ──
MODEL_NAME = "llama-3.3-70b-versatile"
client = Groq(api_key=GROQ_API_KEY)

# ── FastAPI app setup ──
app = FastAPI(title="Bengali Q&A Generator")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("output", exist_ok=True)


# ─────────────────────────────────────────────
#  THE BENGALI Q&A PROMPT
# ─────────────────────────────────────────────
BENGALI_PROMPT_TEMPLATE = """
তুমি একজন বিশেষজ্ঞ বাংলা শিক্ষক এবং ডেটাসেট নির্মাতা।

নিচের বাংলা টেক্সট থেকে ১০ জোড়া প্রশ্ন-উত্তর তৈরি করো।
প্রতিটি প্রশ্ন-উত্তর নিচের ৩টি ভিন্ন ধরনের হতে হবে (সব মিলিয়ে ১০টি):
১. তথ্যমূলক (Factual) — সরাসরি তথ্য জিজ্ঞেস করে
২. কারণমূলক (Causal) — কেন/কীভাবে জিজ্ঞেস করে
৩. যুক্তিমূলক (Reasoning) — বিশ্লেষণ বা সিদ্ধান্ত জিজ্ঞেস করে

উত্তর অবশ্যই বাংলায় দিতে হবে।

এই EXACT ফরম্যাটে লিখবে (কোনো অতিরিক্ত টেক্সট নয়):

Instruction: [বাংলায় প্রশ্ন]
Response: [বাংলায় উত্তর]
Category: Factual

Instruction: [বাংলায় প্রশ্ন]
Response: [বাংলায় উত্তর]
Category: Causal

Instruction: [বাংলায় প্রশ্ন]
Response: [বাংলায় উত্তর]
Category: Reasoning

টেক্সট:
\"\"\"
{text}
\"\"\"
"""


# ─────────────────────────────────────────────
#  PARSE GEMINI OUTPUT INTO ROWS
# ─────────────────────────────────────────────
def parse_qa_response(response_text: str) -> list:
    rows = []
    blocks = re.split(r'\n(?=Instruction:)', response_text.strip())

    for block in blocks:
        instruction_match = re.search(r'Instruction:\s*(.+?)(?=Response:|$)', block, re.DOTALL)
        response_match    = re.search(r'Response:\s*(.+?)(?=Category:|$)', block, re.DOTALL)
        category_match    = re.search(r'Category:\s*(\w+)', block)

        if instruction_match and response_match and category_match:
            rows.append({
                "Instruction": instruction_match.group(1).strip(),
                "Response":    response_match.group(1).strip(),
                "Category":    category_match.group(1).strip(),
            })

    return rows


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate(
    request: Request,
    file: UploadFile = File(None),
    url: str = Form(None),
):
    all_rows = []
    source_name = "output"

    try:
        if file and file.filename:
            file_bytes = await file.read()
            print(f"[app] Received PDF: {file.filename} ({len(file_bytes)} bytes)")
            text = extract_text_from_pdf(file_bytes)
            source_name = file.filename.replace(".pdf", "")
        elif url and url.strip():
            print(f"[app] Received URL: {url.strip()}")
            text = extract_text_from_url(url.strip())
            source_name = "url_content"
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Please upload a PDF or enter a URL."}
            )

        print(f"[app] Total text length: {len(text)} characters")

        batches = split_into_batches(text, chars_per_batch=2500)
        total_batches = len(batches)

        for i, batch_text in enumerate(batches):
            print(f"[app] Processing batch {i+1}/{total_batches}…")
            prompt = BENGALI_PROMPT_TEMPLATE.format(text=batch_text)

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=4096,
                )
                reply = response.choices[0].message.content
                parsed = parse_qa_response(reply)
                all_rows.extend(parsed)
                print(f"[app] Batch {i+1}: got {len(parsed)} Q&A pairs ✓")
            except Exception as e:
                print(f"[app] ⚠️ Batch {i+1} failed: {e}")

            if i < total_batches - 1:
                time.sleep(2)

        if not all_rows:
            return JSONResponse(
                status_code=500,
                content={"error": "Model returned no Q&A pairs. Try a different text."}
            )

        # Save CSV to output/ folder
        filename = f"{source_name}_bengali_qa.csv"
        filepath = os.path.join("output", filename)

        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["Instruction", "Response", "Category"],
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"[app] ✅ Done! Total Q&A pairs: {len(all_rows)} → saved to {filepath}")

        return JSONResponse(content={
            "success": True,
            "filename": filename,
            "total_pairs": len(all_rows),
        })

    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Server error: {str(e)}"})


@app.get("/download/{filename}")
async def download(filename: str):
    """Serve a generated CSV file from the output/ folder."""
    filepath = os.path.join("output", filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "File not found."})
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="text/csv",
    )


@app.get("/health")
async def health():
    return {"status": "running", "model": MODEL_NAME}
