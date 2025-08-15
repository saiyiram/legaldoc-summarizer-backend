import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

# Use Hugging Face secret for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded."}
    )

def extract_text_from_pdf(file_bytes):
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return "\n".join([page.get_text() for page in doc])

def chunk_text(text, max_tokens=1500):
    max_chars = max_tokens * 4
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

async def summarize_with_gpt5_nano(text: str) -> str:
    try:
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system",
                 "content": "You are a legal document assistant. Summarize contracts clearly, under 600 words."},
                {"role": "user", "content": text}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

async def summarize_large_text(text):
    chunks = chunk_text(text)
    summaries = [await summarize_with_gpt5_nano(c) for c in chunks]
    if len(summaries) > 1:
        return await summarize_with_gpt5_nano(" ".join(summaries))
    return summaries[0]

@app.post("/upload_pdf/")
@limiter.limit("5/minute")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    if file.content_type.lower() != "application/pdf":
        return {"error": "Only PDF files are supported."}
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    summary = await summarize_large_text(text)
    return {"summary": summary}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
