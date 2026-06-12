
import streamlit as st


st.set_page_config(
    page_title="DocMind – Multi-Source RAG",
    page_icon="",
    layout="wide"
)


from Backend.document_loader import load_pdf, load_url, load_text
from Backend.rag_pipeline import create_vector_store, build_rag_chain

st.markdown(
    "<h2 style='text-align: center;'>DocMind</h2>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align: center; color: gray;">
        Multi-source RAG chatbot — load PDFs, URLs, or plain text, then ask anything.
    </div>
    """,
    unsafe_allow_html=True
)



# SESSION STATE


defaults = {
    "messages":       [],    
    "vector_store":   None,  
    "rag_chain":      None,  
    "loaded_sources": [],    
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value



# SIDEBAR — Document Loader


with st.sidebar:
    st.header("Load Your Documents")
    st.markdown("Add one or more sources below, then click **Build Knowledge Base**.")

    # ── Source 1: PDF Upload 
    st.subheader("Upload PDF(s)")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once. Each page becomes a separate chunk."
    )

    # ── Source 2: URL 
    st.subheader("Add a Webpage URL")
    url_input = st.text_input(
        "Paste a URL",
        placeholder="https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    )

    # ── Source 3: Raw Text 
    st.subheader("Paste Text")
    text_input = st.text_area(
        "Paste any text content",
        placeholder="Paste notes, articles, documentation...",
        height=140
    )

    st.divider()

    # ── Build Button 
    build_clicked = st.button("Build Knowledge Base", use_container_width=True)

    if build_clicked:
        all_docs    = []
        source_log  = []
        errors      = []

        # Process PDFs
        for file in uploaded_files:
            try:
                docs = load_pdf(file)
                all_docs.extend(docs)
                source_log.append(f"{file.name}  ({len(docs)} pages loaded)")
            except Exception as e:
                errors.append(f"PDF — {file.name}: {e}")

        # Process URL
        if url_input.strip():
            try:
                docs = load_url(url_input.strip())
                all_docs.extend(docs)
                source_log.append(f"{url_input.strip()}")
            except Exception as e:
                errors.append(f"URL: {e}")

        # Process raw text
        if text_input.strip():
            try:
                docs = load_text(text_input.strip(), label="Pasted Text")
                all_docs.extend(docs)
                source_log.append(" Pasted Text")
            except Exception as e:
                errors.append(f"Text: {e}")

        # Show any errors that occurred during loading
        for err in errors:
            st.error(f"{err}")

        if not all_docs:
            st.warning("No content loaded. Please add at least one source.")
        else:
            with st.spinner(f"Embedding {len(all_docs)} document pages into FAISS..."):
                try:
                    vector_store = create_vector_store(all_docs)
                    rag_chain    = build_rag_chain(vector_store)

                    st.session_state.vector_store   = vector_store
                    st.session_state.rag_chain      = rag_chain
                    st.session_state.loaded_sources = source_log
                    st.session_state.messages       = []   # clear chat for new session

                    st.success(f"Ready! Indexed {len(all_docs)} document chunks.")

                except Exception as e:
                    st.error(f"Failed to build knowledge base: {e}")

    # ── Show loaded sources ───────────────────────────────────
    if st.session_state.loaded_sources:
        st.divider()
        st.subheader("Currently Loaded")
        for src in st.session_state.loaded_sources:
            st.markdown(f"- {src}")

    # ── Clear button ──────────────────────────────────────────
    if st.session_state.rag_chain:
        st.divider()
        if st.button("Clear Everything", use_container_width=True):
            for key in defaults:
                st.session_state[key] = defaults[key]
            st.rerun()



# MAIN AREA — Chat Interface



if not st.session_state.rag_chain:
    st.markdown(
        """
        <div style="text-align:center;">
             Use the sidebar to load your documents, then click
            <b>Build Knowledge Base</b> to start chatting.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box (appears at bottom of page)
question = st.chat_input("Ask anything about your documents...")

if question:

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    # Generate and stream the assistant response
    with st.chat_message("assistant"):
        full_response = ""
        placeholder   = st.empty()

        try:
            
            for token in st.session_state.rag_chain.stream(question):
                full_response += token
                placeholder.markdown(full_response + "▌")  

            # Final render without the cursor
            placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"Something went wrong: {e}"
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
