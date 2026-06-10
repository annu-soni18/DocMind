
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
