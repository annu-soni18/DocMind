"""
Prompt/prompt_template.py
-------------------------
Defines the prompt template sent to the LLM.

INTERVIEW POINT — How does your prompt prevent hallucination?
    The instruction "ONLY use the information in the context below"
    constrains the LLM to the retrieved chunks. The fallback sentence
    "I couldn't find this in the provided documents" gives it a safe
    exit instead of making something up.

INTERVIEW POINT — How do citations work in the prompt?
    By the time this prompt is filled, {context} already contains
    chunks pre-labelled with [Source: filename] (done in rag_pipeline.py).
    The prompt then instructs the LLM to reference those source labels.
    This two-step approach (label in context + instruct in prompt)
    produces reliable citations without any post-processing.
"""

from langchain_core.prompts import PromptTemplate


def get_prompt() -> PromptTemplate:
    template = """You are a helpful assistant that answers questions strictly based on the provided documents.

Instructions:
- Use ONLY the information from the context below to answer.
- Always mention the source (e.g. "According to [Source: filename]...").
- If the answer is not in the context, say: "I couldn't find this in the provided documents."
- Be concise and clear. Do not repeat the question.

Context:
{context}

Question:
{question}

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
