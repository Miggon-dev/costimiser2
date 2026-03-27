# text_summarization.py

from typing import Optional


def summarize_answer_text(
    text: str,
    language: str = "English",
    style: str = "concise",
) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""

    prompt = f"""
You are summarizing an industrial analytics explanation for a mill user.

Requirements:
- Write in {language}
- Style: {style}
- Preserve key numbers and conclusions
- Preserve warnings and uncertainties
- Preserve savings estimates if present
- Preserve review-before-action flags if present
- Do not invent facts
- Keep it shorter and clearer than the original

Text to summarize:
{text}
""".strip()

    import knowledge_retrieval as rag

    response = rag.ask(prompt)

    if response is None:
        return text

    if isinstance(response, dict):
        if "answer" in response and response["answer"] is not None:
            return str(response["answer"]).strip()
        if "text" in response and response["text"] is not None:
            return str(response["text"]).strip()

    return str(response).strip()