"""
llm.py — Centralised LLM factory (Groq / LPU backend).
"""
from langchain_groq import ChatGroq
from src.core.config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE


def get_llm(temperature: float = LLM_TEMPERATURE, json_mode: bool = False) -> ChatGroq:
    """
    Returns a ChatGroq instance.
    Pass json_mode=True for agents that need strict JSON output
    (binds response_format={"type": "json_object"}).
    """
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=temperature,
        max_retries=3,
    )
    if json_mode:
        llm = llm.bind(response_format={"type": "json_object"})
    return llm
