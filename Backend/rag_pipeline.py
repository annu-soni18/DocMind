
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
    
    embeddings = load_embeddings()
    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError(
            "No content could be extracted from the provided sources. "
            "Please check that your PDFs contain text (not just images)."
        )

    
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ──────────────────────────────────────────────────────────────
# STEP 3: Format retrieved chunks for the prompt
# ──────────────────────────────────────────────────────────────

def format_docs_with_sources(docs: list[Document]) -> str:
   
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

    
    return "\n\n---\n\n".join(formatted)


# ──────────────────────────────────────────────────────────────
# STEP 4: Build the full RAG chain
# ──────────────────────────────────────────────────────────────

def build_rag_chain(vector_store: FAISS):
    
    llm      = load_llm()
    prompt   = get_prompt()

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":       4,    
            "fetch_k": 10    
        }
    )

   
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
