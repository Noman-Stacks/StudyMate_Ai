# StudyMate AI

StudyMate AI is a beginner-friendly AI study assistant. Students upload an
educational PDF and use grounded AI tools to ask questions, generate summaries,
and practice with multiple-choice quizzes.

The application is designed for Streamlit Community Cloud and keeps document
processing in memory. It does not use a database or persist uploaded PDFs.

## Features

- Secure PDF upload with a 10 MB limit and password/corruption handling
- Page-aware text extraction and overlapping 500-word chunks
- Cached Sentence Transformers embeddings using `all-MiniLM-L6-v2`
- In-memory FAISS semantic search with cosine similarity
- RAG question answering with Gemini, three retrieved chunks, and source pages
- Short, medium, and detailed grounded summaries
- Validated 5-, 10-, and 15-question quizzes with scoring and explanations
- Four-page navigation: Document Upload, Ask AI, Summarize, Generate Quiz
- Session-state caching to avoid repeated extraction, embedding, and indexing

## Technology stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Google Gemini API
- python-dotenv

## Project structure

```text
StudyMate_Ai/
├── app.py                    # Streamlit application entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Template for local API credentials
├── .gitignore                # Files excluded from Git
├── README.md                 # Project documentation
├── src/
│   ├── __init__.py           # Marks src as a Python package
│   ├── config.py             # Environment-based configuration
│   └── ui.py                 # Streamlit interface
└── modules/
    ├── __init__.py           # Marks modules as a Python package
    ├── pdf_processor.py      # PDF validation and text extraction
    ├── text_chunker.py       # Page-aware overlapping text chunks
    ├── embeddings.py         # Sentence Transformer embeddings
    ├── vector_store.py       # FAISS semantic search
    ├── ai_assistant.py       # Gemini RAG question answering
    ├── summarizer.py         # Section-based Gemini summarization
    ├── quiz_generator.py     # Structured Gemini quiz generation
    └── config.py             # Environment-based Gemini configuration
```

## How it works

1. A PDF is uploaded and validated in memory.
2. PyMuPDF extracts text page by page.
3. Text is split into chunks with page metadata.
4. Sentence Transformers creates normalized semantic vectors.
5. FAISS indexes the vectors for similarity search.
6. Ask AI retrieves the three closest chunks and sends only that context to
   Gemini.
7. Summarization uses section-level Gemini calls followed by synthesis.
8. Quiz generation requests strict JSON and validates every question before
   display.

## RAG architecture

Retrieval-Augmented Generation separates retrieval from generation. The
student's question is embedded with the same model used for document chunks.
FAISS retrieves the five most relevant chunks using inner product on normalized
vectors, which is equivalent to cosine similarity. Those chunks become the
only document context supplied to Gemini. This grounds answers in the uploaded
PDF and lets the interface show source pages.

## Installation

1. Create and activate a virtual environment:

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Configure the Gemini API credential locally:

   **Windows PowerShell**

   ```powershell
   Copy-Item .env.example .env
   ```

   **macOS/Linux**

   ```bash
   cp .env.example .env
   ```

   Then replace the placeholder value in `.env` with your Gemini API key.
   Never commit `.env` or expose the key in source code or the frontend.

## Run locally

```bash
streamlit run app.py
```

The app opens in a browser. Upload a PDF from the sidebar, then choose a
feature from the navigation. The same document remains available across pages.

## Deployment on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, create a new app and select the repository,
   branch, and `app.py` as the main file.
3. Open the app's **Settings > Secrets** and add:

   ```toml
   GEMINI_API_KEY = "your_gemini_api_key"
   GEMINI_MODEL = "gemini-3.8-flash"
   ```

4. Deploy or reboot the app.

The application reads secrets from Streamlit secrets first when no environment
variable is present. Secrets are never shown in the UI. Do not upload `.env`
or put credentials in GitHub.

## How summarization works

The summarizer groups page text into manageable sections of up to 2,500 words.
Gemini creates a grounded summary for each section, preserving scientific
terms, definitions, and concepts. A final Gemini pass combines those section
summaries into one coherent short, medium, or detailed summary. Every prompt
instructs Gemini to use only the supplied document text and not invent facts.

## How quizzes work

Gemini receives the extracted PDF text and is required to return strict JSON.
The application validates the exact question count, four distinct options,
one valid correct-option index, and a non-empty explanation before displaying
the quiz. Students answer the questions in a form, submit once, and then see
their score, correct answers, and explanations. Questions are generated only
from the supplied document context.

## Screenshots

Screenshots are not included yet. Before publishing a deployment, capture:

- `Document Upload` with a loaded PDF
- `Ask AI` showing an answer and source pages
- `Summarize` showing a generated summary
- `Generate Quiz` showing the score and explanations

## Known limitations

- Scanned PDFs require OCR; OCR is not included.
- Only one PDF is active per browser session.
- Uploaded files, FAISS indexes, and generated results are stored in memory and
  are lost when the session or Cloud app restarts.
- Gemini and Sentence Transformers require network access and can be affected
  by model download time, API quotas, or service availability.
- Very long documents are bounded by the summarizer and quiz context limits.
- Generated answers, summaries, and questions can still contain model errors;
  source grounding and validation reduce but do not eliminate this risk.

## Test the application

1. Install dependencies and start the app with `streamlit run app.py`.
2. Upload a normal text-based PDF and confirm the filename, page count, and
   extracted-text preview are shown.
3. Upload a scanned/image-only PDF and confirm the OCR warning is displayed.
4. Try an oversized file, a non-PDF file, a corrupted PDF, or a
   password-protected PDF and confirm a helpful error is shown.
5. Upload a text-based PDF with enough content to confirm the generated chunk
   count and embedding vector dimensions are displayed.
6. On the first embedding run, allow Sentence Transformers to download
   `all-MiniLM-L6-v2`. Later runs reuse the cached model.
7. Navigate to `Ask AI`, ask a question, and confirm the answer includes source
   PDF pages.
8. Navigate to `Summarize`, choose a length, and generate a summary.
9. Navigate to `Generate Quiz`, choose 5, 10, or 15 questions, answer every
   question, and submit to verify scoring and explanations.
10. Upload a second PDF and confirm the index, chat, summary, and quiz state are
    rebuilt for the new document.

Run the automated tests with:

```bash
python -m unittest discover tests
```

To run the live Gemini connection check without printing the API key:

```bash
python scripts/test_gemini_connection.py
```

## Main AI concepts

- **Embeddings:** numerical representations of text meaning.
- **FAISS:** fast nearest-neighbor search over embedding vectors.
- **RAG:** retrieve relevant source text before generating an answer.
- **Grounding:** prompts restrict Gemini to supplied document context.
- **Structured generation:** quiz JSON is parsed and validated before display.
