"""Streamlit interface for StudyMate AI."""

from dataclasses import dataclass
from hashlib import sha256

import streamlit as st

from modules.ai_assistant import (
    AIAssistantError,
    GeminiAPIError,
    GeminiConfigurationError,
    GeminiRateLimitError,
    answer_question,
)
from modules.embeddings import EmbeddingError, EmbeddedChunks, embed_chunks
from modules.pdf_processor import (
    MAX_FILE_SIZE_MB,
    CorruptedPDFError,
    PDFExtractionResult,
    PDFProcessingError,
    PasswordProtectedPDFError,
    extract_pdf_text,
)
from modules.quiz_generator import (
    QUESTION_COUNTS,
    QuizAPIError,
    QuizConfigurationError,
    QuizGeneratorError,
    QuizRateLimitError,
    QuizValidationError,
    generate_quiz,
)
from modules.summarizer import (
    SUMMARY_LENGTHS,
    SummaryAPIError,
    SummaryConfigurationError,
    SummaryRateLimitError,
    SummarizerError,
    summarize_document,
)
from modules.text_chunker import ChunkingError, TextChunk, create_text_chunks
from modules.vector_store import FAISSVectorStore, VectorStoreError


@dataclass(frozen=True)
class PreparedDocument:
    """All derived document data reused across Streamlit reruns."""

    fingerprint: str
    filename: str
    extraction: PDFExtractionResult
    chunks: tuple[TextChunk, ...]
    embeddings: EmbeddedChunks
    vector_store: FAISSVectorStore


def render_app() -> None:
    """Render the complete StudyMate AI application."""
    st.set_page_config(
        page_title="StudyMate AI",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    _render_sidebar()

    page = st.session_state.get("active_page", "Document Upload")
    document = st.session_state.get("prepared_document")

    if page == "Document Upload":
        _render_upload_page(document)
    elif page == "Ask AI":
        _render_ask_page(document)
    elif page == "Summarize":
        _render_summary_page(document)
    else:
        _render_quiz_page(document)


def _render_sidebar() -> None:
    """Render navigation and the shared PDF uploader."""
    with st.sidebar:
        st.markdown("# StudyMate AI")
        st.caption("Learn from your documents with grounded AI.")
        page = st.radio(
            "Navigate",
            ("Document Upload", "Ask AI", "Summarize", "Generate Quiz"),
            key="active_page",
        )
        st.divider()
        st.markdown("### Document")
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            accept_multiple_files=False,
            help=f"PDF files only, up to {MAX_FILE_SIZE_MB} MB.",
        )
        if uploaded_file is not None:
            _prepare_uploaded_file(uploaded_file.name, uploaded_file.getvalue())

        document = st.session_state.get("prepared_document")
        if document is None:
            st.caption("No document loaded.")
        else:
            st.success(document.filename)
            st.caption(
                f"{document.extraction.page_count} pages | "
                f"{len(document.chunks)} chunks"
            )
        if page != st.session_state.get("active_page"):
            st.session_state["active_page"] = page


def _prepare_uploaded_file(filename: str, file_bytes: bytes) -> None:
    """Prepare a new PDF once and reuse derived data for later reruns."""
    fingerprint = sha256(file_bytes).hexdigest()
    current = st.session_state.get("prepared_document")
    if isinstance(current, PreparedDocument) and current.fingerprint == fingerprint:
        return

    try:
        with st.spinner("Reading PDF and building semantic index..."):
            extraction = extract_pdf_text(file_bytes)
            if not extraction.has_extractable_text:
                raise PDFProcessingError(
                    "No extractable text was found. Scanned PDFs require OCR."
                )
            chunks = create_text_chunks(extraction, filename)
            embeddings = embed_chunks(chunks)
            vector_store = FAISSVectorStore(embeddings.model_name)
            vector_store.rebuild(chunks, embeddings.vectors)
    except (
        ValueError,
        ChunkingError,
        CorruptedPDFError,
        PasswordProtectedPDFError,
        PDFProcessingError,
        EmbeddingError,
        VectorStoreError,
    ) as error:
        st.error(str(error))
        st.session_state.pop("prepared_document", None)
        return

    st.session_state["prepared_document"] = PreparedDocument(
        fingerprint=fingerprint,
        filename=filename,
        extraction=extraction,
        chunks=chunks,
        embeddings=embeddings,
        vector_store=vector_store,
    )
    st.session_state["chat_history"] = []
    st.session_state.pop("quiz", None)
    st.session_state.pop("quiz_answers", None)
    st.session_state.pop("quiz_submitted", None)


def _render_upload_page(document: PreparedDocument | None) -> None:
    """Render the landing page and document status."""
    st.title("StudyMate AI")
    st.subheader("Your personal AI study workspace")
    st.write(
        "Upload an educational PDF from the sidebar, then ask questions, "
        "generate a grounded summary, or test your knowledge."
    )
    cards = st.columns(3)
    cards[0].metric("PDF limit", f"{MAX_FILE_SIZE_MB} MB")
    cards[1].metric("Embedding model", "MiniLM-L6-v2")
    cards[2].metric("Search results", "Top 3")

    if document is None:
        st.info("Upload a PDF from the sidebar to get started.")
        st.markdown(
            "### What you can do\n"
            "- **Ask AI:** Get answers grounded in your PDF\n"
            "- **Summarize:** Create short, medium, or detailed summaries\n"
            "- **Generate Quiz:** Practice with document-based questions"
        )
        return

    st.success(f"{document.filename} is ready.")
    st.write(
        f"Extracted **{document.extraction.page_count} pages** and created "
        f"**{len(document.chunks)} searchable chunks**."
    )
    st.info("Use the sidebar navigation to choose a study activity.")


def _render_ask_page(document: PreparedDocument | None) -> None:
    """Render the grounded question-answering page."""
    st.title("Ask AI")
    st.caption("Answers are generated from the uploaded PDF and include sources.")
    if document is None:
        st.info("Upload a PDF from the sidebar before asking questions.")
        return

    history = st.session_state.setdefault("chat_history", [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDF")
    if not question:
        return

    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the document and consulting Gemini..."):
            try:
                response = answer_question(document.vector_store, question)
            except (
                GeminiConfigurationError,
                GeminiRateLimitError,
                GeminiAPIError,
                AIAssistantError,
                VectorStoreError,
                ValueError,
            ) as error:
                st.error(str(error))
                return
        pages = sorted(
            {
                page
                for source in response.sources
                for page in source.chunk.page_numbers
            }
        )
        source_text = ", ".join(map(str, pages))
        st.markdown(response.answer)
        st.caption(f"Sources: PDF pages {source_text}")
        history.append(
            {
                "role": "assistant",
                "content": f"{response.answer}\n\n*Sources: PDF pages {source_text}*",
            }
        )


def _render_summary_page(document: PreparedDocument | None) -> None:
    """Render the document summarization page."""
    st.title("Summarize")
    st.caption("Create a grounded summary without adding facts outside the PDF.")
    if document is None:
        st.info("Upload a PDF from the sidebar before generating a summary.")
        return

    summary_length = st.radio(
        "Summary length",
        SUMMARY_LENGTHS,
        horizontal=True,
        format_func=lambda value: f"{value} summary",
    )
    if not st.button("Generate summary", type="primary"):
        return

    with st.spinner("Summarizing document sections..."):
        try:
            result = summarize_document(document.extraction, summary_length)
        except (
            SummaryConfigurationError,
            SummaryRateLimitError,
            SummaryAPIError,
            SummarizerError,
            ValueError,
        ) as error:
            st.error(str(error))
            return
    st.success(f"Summary generated from {result.section_count} section(s).")
    st.markdown(result.summary)


def _render_quiz_page(document: PreparedDocument | None) -> None:
    """Render quiz generation, submission, scoring, and explanations."""
    st.title("Generate Quiz")
    st.caption("Test yourself using questions generated only from your PDF.")
    if document is None:
        st.info("Upload a PDF from the sidebar before generating a quiz.")
        return

    question_count = st.selectbox(
        "Number of questions",
        QUESTION_COUNTS,
        format_func=lambda value: f"{value} questions",
    )
    if st.button("Generate quiz", type="primary"):
        with st.spinner("Creating a validated quiz..."):
            try:
                quiz = generate_quiz(document.extraction, question_count)
            except (
                QuizConfigurationError,
                QuizRateLimitError,
                QuizAPIError,
                QuizGeneratorError,
                QuizValidationError,
                ValueError,
            ) as error:
                st.error(str(error))
                return
        st.session_state["quiz"] = quiz
        st.session_state["quiz_submitted"] = False
        st.session_state.pop("quiz_answers", None)

    quiz = st.session_state.get("quiz")
    if quiz is None:
        st.info("Choose a question count and generate a quiz to begin.")
        return

    with st.form("quiz_form"):
        answers: dict[int, int] = {}
        for index, quiz_question in enumerate(quiz.questions):
            st.markdown(f"**{index + 1}. {quiz_question.question}**")
            selected = st.radio(
                "Choose one answer:",
                quiz_question.options,
                index=None,
                key=f"quiz_answer_{index}",
                label_visibility="collapsed",
            )
            if selected is not None:
                answers[index] = quiz_question.options.index(selected)
        submitted = st.form_submit_button("Submit answers", type="primary")

    if submitted:
        if len(answers) != len(quiz.questions):
            st.warning("Please answer every question before submitting.")
            return
        st.session_state["quiz_answers"] = answers
        st.session_state["quiz_submitted"] = True

    if not st.session_state.get("quiz_submitted"):
        return

    submitted_answers = st.session_state["quiz_answers"]
    score = sum(
        submitted_answers[index] == question.correct_index
        for index, question in enumerate(quiz.questions)
    )
    st.success(f"Your score: {score} / {len(quiz.questions)}")
    for index, question in enumerate(quiz.questions):
        is_correct = submitted_answers[index] == question.correct_index
        label = "Correct" if is_correct else "Incorrect"
        st.markdown(
            f"**Question {index + 1}: {label}**  \n"
            f"Correct answer: {question.options[question.correct_index]}"
        )
        st.caption(question.explanation)


def _inject_styles() -> None:
    """Apply a small consistent theme without hiding Streamlit controls."""
    st.markdown(
        """
        <style>
        .block-container { max-width: 1100px; padding-top: 2rem; }
        [data-testid="stMetric"] {
            background: rgba(99, 102, 241, 0.08);
            border-radius: 0.75rem;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
