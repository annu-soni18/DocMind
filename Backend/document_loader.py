
import fitz                          
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document   


# 1. PDF LOADER


def load_pdf(uploaded_file) -> list[Document]:
   
    documents = []

    # Read raw bytes from the Streamlit uploader
    pdf_bytes = uploaded_file.read()

    # fitz.open() accepts raw bytes with filetype hint
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text()          

        if text.strip():                
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": uploaded_file.name,   
                    "page":   page_num + 1          
                }
            ))

    pdf.close()
    return documents



# 2. URL LOADER


def load_url(url: str) -> list[Document]:
    
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



# 3. RAW TEXT LOADER


def load_text(text: str, label: str = "Pasted Text") -> list[Document]:
   
    if not text.strip():
        raise ValueError("Text input is empty. Please paste some content.")

    return [Document(
        page_content=text,
        metadata={"source": label}
    )]
