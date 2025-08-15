import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Rate limit error handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please wait before trying again."}
    )

# PDF text extraction
def extract_text_from_pdf(file_bytes):
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return "\n".join([page.get_text() for page in doc])

# Chunking text (approximate 1 token ≈ 4 chars)
def chunk_text(text, max_tokens=1500):
    max_chars = max_tokens * 4
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

# GPT-5 Nano summarization
async def summarize_with_gpt5_nano(text: str) -> str:
    try:
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal document assistant that summarizes contracts and agreements "
                        "in simple, easy-to-understand language. Avoid legal jargon unless necessary."
                        "Keep the summary concise and under 600 words."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
                Please analyze the following legal document and return your answer in **Markdown** format, 
                with headings, bullet points, and sub-points, so it can be displayed cleanly on a webpage:

                {text}

                Your response should include:
                1. A plain-language summary of the document.
                2. Key clauses, obligations, and unusual terms.
                3. Any potential red flags or risks.
                4. A note that the user can ask specific follow-up questions.
                5. Suggest consulting a legal professional if necessary.

                Use:
                - `###` for main headings
                - `-` for bullet points
                - `**bold**` for important terms
                """
}

            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during summarization: {str(e)}"

# Summarize large text
async def summarize_large_text(text):
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        summaries.append(await summarize_with_gpt5_nano(chunk))

    if len(summaries) > 1:
        return await summarize_with_gpt5_nano(" ".join(summaries))
    return summaries[0]

# Upload PDF endpoint
@app.post("/upload_pdf/")
@limiter.limit("5/minute")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    if file.content_type.lower() != "application/pdf":
        return {"error": "Only PDF files are supported."}

    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    summary = await summarize_large_text(text)
    return {"summary": summary}
