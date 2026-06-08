"""
Backend/rag_pipeline.py
-----------------------
Two main functions:
  1. create_vector_store(documents) → chunks docs → embeds → FAISS index
  2. build_rag_chain(vector_store)  → full retrieval + LLM chain

HOW THE PIPELINE WORKS (explain this in interviews):

  User question
       │
       ▼
  Retriever (FAISS MMR search)
       │  finds top-4 relevant chunks
       ▼
  format_docs_with_sources()
       │  adds [Source: filename] label to each chunk
       ▼
  Prompt template
       │  fills {context} and {question} placeholders
       ▼
  Groq LLaMA3 LLM
       │  generates answer, instructed to cite sources
       ▼
  StrOutputParser
       │  converts LLM output to plain string
       ▼
  Streamed token-by-token to the Streamlit UI
"""

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from Models.llm_model import load_embeddings, load_llm
from Prompt.prompt_template import get_prompt


# ──────────────────────────────────────────────────────────────
# STEP 1: Chunk documents
# ──────────────────────────────────────────────────────────────

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits large documents into smaller overlapping chunks.

    INTERVIEW POINT — Why do we chunk?
        LLMs have a limited context window. If we send a 50-page PDF
        as one string it won't fit. We split it into ~1000-char pieces.

    INTERVIEW POINT — Why chunk_overlap=200?
        Imagine a sentence starts on chunk 3 and ends on chunk 4.
        Without overlap, the meaning is split across two chunks and
        neither chunk alone makes sense. With 200-char overlap, the
        end of chunk 3 is repeated at the start of chunk 4 — so
        retrieved chunks always contain complete thoughts.

    INTERVIEW POINT — Why RecursiveCharacterTextSplitter?
        It tries to split on paragraph breaks first (\n\n), then
        line breaks (\n), then sentences (". "), then words (" ").
        This keeps semantically related text together as much as possible.

    NOTE: LangChain automatically copies metadata (source, page) from
    the original Document into every chunk it creates.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ──────────────────────────────────────────────────────────────
# STEP 2: Build FAISS vector store
# ──────────────────────────────────────────────────────────────

def create_vector_store(documents: list[Document]) -> FAISS:
    """
    Chunks all documents and stores their embeddings in a FAISS index.

    Args:
        documents: List of Document objects from document_loader.py

    Returns:
        A FAISS vector store ready for similarity search.

    INTERVIEW POINT — What is FAISS?
        Facebook AI Similarity Search. It stores embedding vectors in RAM
        and performs extremely fast cosine similarity search.
        When a user asks a question, the question is also embedded into
        a vector, and FAISS finds the chunks whose vectors are closest
        to the question vector.

    INTERVIEW POINT — Why FAISS over ChromaDB?
        FAISS is in-memory, zero setup, very fast for small-to-medium
        datasets. Perfect for a demo/portfolio app where data is
        session-based (no need for persistence).
        ChromaDB persists to disk — better if you want the index to
        survive app restarts.
    """
    embeddings = load_embeddings()
    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError(
            "No content could be extracted from the provided sources. "
            "Please check that your PDFs contain text (not just images)."
        )

    # FAISS.from_documents() calls the embeddings model for every chunk
    # and builds the search index in one shot
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ──────────────────────────────────────────────────────────────
# STEP 3: Format retrieved chunks for the prompt
# ──────────────────────────────────────────────────────────────

def format_docs_with_sources(docs: list[Document]) -> str:
    """
    Converts a list of retrieved Document chunks into a single
    formatted string that includes the source label for each chunk.

    INTERVIEW POINT — How do citations work?
        Step 1: Each chunk has metadata["source"] from when it was loaded.
        Step 2: This function prepends [Source: filename] before each chunk.
        Step 3: The prompt tells the LLM "mention the source in your answer."
        Result: The LLM naturally says "According to [Source: paper.pdf]..."

    This is called inside the RAG chain via RunnableLambda.
    """
    formatted = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        page   = doc.metadata.get("page", "")

        # If it came from a PDF, show page number too
        if page:
            source_label = f"{source} — page {page}"
        else:
            source_label = source

        formatted.append(
            f"[Source: {source_label}]\n{doc.page_content}"
        )

    # Separate chunks with a clear divider so the LLM can distinguish them
    return "\n\n---\n\n".join(formatted)


# ──────────────────────────────────────────────────────────────
# STEP 4: Build the full RAG chain
# ──────────────────────────────────────────────────────────────

def build_rag_chain(vector_store: FAISS):
    """
    Builds and returns the complete LCEL (LangChain Expression Language) chain:

        retriever | format_context | prompt | llm | output_parser

    INTERVIEW POINT — What is LCEL?
        LangChain Expression Language. The | (pipe) operator chains
        components together. Output of one step becomes input of the next.
        It supports .stream() out of the box — giving token-by-token
        streaming to the UI without extra code.

    INTERVIEW POINT — What is MMR?
        Maximal Marginal Relevance. Instead of returning the top-4 most
        similar chunks (which might all say the same thing), MMR picks
        chunks that are both relevant AND diverse from each other.
        fetch_k=10 means: retrieve 10 candidates, then pick the 4 most
        diverse ones. This gives the LLM richer context.
    """
    llm      = load_llm()
    prompt   = get_prompt()

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":       4,    # final number of chunks returned to the LLM
            "fetch_k": 10    # candidates considered before MMR filtering
        }
    )

    # Build the chain using LCEL pipe syntax
    # RunnablePassthrough() passes the user question through unchanged
    # RunnableLambda() wraps our custom format function as a chain step
    rag_chain = (
        {
            "context":  retriever | RunnableLambda(format_docs_with_sources),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
