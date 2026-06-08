
import fitz                          # PyMuPDF — installed as 'pymupdf', imported as 'fitz'
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document   # standard LangChain document object


# ──────────────────────────────────────────────────────────────
# 1. PDF LOADER
# ──────────────────────────────────────────────────────────────

def load_pdf(uploaded_file) -> list[Document]:
    """
    Reads a Streamlit-uploaded PDF and returns one Document per page.

    Args:
        uploaded_file: Streamlit UploadedFile object from st.file_uploader()

    Returns:
        List of Document objects, one per non-blank page.

    INTERVIEW POINT:
        We create one Document per page (not the whole file as one Document)
        because it makes chunking more natural — page boundaries are
        logical break points in a document.
        Metadata stores both filename and page number for precise citations.
    """
    documents = []

    # Read raw bytes from the Streamlit uploader
    pdf_bytes = uploaded_file.read()

    # fitz.open() accepts raw bytes with filetype hint
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text()          # extract plain text from this page

        if text.strip():                # skip blank/image-only pages
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": uploaded_file.name,   # e.g. "machine_learning.pdf"
                    "page":   page_num + 1          # 1-indexed for human readability
                }
            ))

    pdf.close()
    return documents


# ──────────────────────────────────────────────────────────────
# 2. URL LOADER
# ──────────────────────────────────────────────────────────────

def load_url(url: str) -> list[Document]:
    """
    Fetches a webpage and extracts readable text, ignoring navigation/scripts.

    Args:
        url: Full URL string (e.g. "https://en.wikipedia.org/wiki/RAG")

    Returns:
        A single-item list with one Document containing the page text.

    INTERVIEW POINT:
        We manually use requests + BeautifulSoup instead of LangChain's
        WebBaseLoader because it gives us full control over:
          - Custom User-Agent headers (some sites block bots)
          - Filtering out nav/footer/script noise
          - Timeout and error handling for deployment on Render
    """
    try:
        headers = {
            # Without this, many websites return 403 Forbidden
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()     # raises for 4xx / 5xx HTTP errors

        # Parse HTML with lxml parser (faster than html.parser, installed via requirements.txt)
        soup = BeautifulSoup(response.text, "lxml")

        # Remove non-content tags before extracting text
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()

        # get_text() joins all remaining text; separator keeps paragraphs readable
        raw_text = soup.get_text(separator="\n", strip=True)

        # Filter lines that are too short to be real content (buttons, labels, etc.)
        lines = [line for line in raw_text.splitlines() if len(line.strip()) > 40]
        clean_text = "\n".join(lines)

        if not clean_text.strip():
            raise ValueError("No readable content found at this URL.")

        return [Document(
            page_content=clean_text,
            metadata={"source": url}
        )]

    except requests.exceptions.Timeout:
        raise ValueError(f"Request timed out. URL took too long to respond: {url}")
    except requests.exceptions.ConnectionError:
        raise ValueError(f"Could not connect to URL. Check the address: {url}")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error {e.response.status_code} for URL: {url}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch URL: {e}")


# ──────────────────────────────────────────────────────────────
# 3. RAW TEXT LOADER
# ──────────────────────────────────────────────────────────────

def load_text(text: str, label: str = "Pasted Text") -> list[Document]:
    """
    Wraps a plain text string into a LangChain Document.

    Args:
        text:  The raw text content.
        label: A human-readable name shown in citations.

    Returns:
        A single-item list with one Document.

    INTERVIEW POINT:
        Even raw text gets wrapped in a Document with metadata.
        This is the same structure as PDF/URL output, so the rest of the
        pipeline (rag_pipeline.py) treats all sources identically.
    """
    if not text.strip():
        raise ValueError("Text input is empty. Please paste some content.")

    return [Document(
        page_content=text,
        metadata={"source": label}
    )]
