"""
LLM generation pipeline with grounded prompting.

Architecture decision: we use an explicit system prompt that instructs the
LLM to answer *only* from the supplied context.  This is the single most
important design choice for RAG faithfulness:

1. It reduces hallucination by removing the model's incentive to rely on
   parametric memory.
2. It makes faithfulness scores from RAGAS more meaningful — a model that
   ignores the context despite this instruction is genuinely unfaithful.
3. The explicit "say you don't know" instruction gives evaluators a signal
   that non-answers are correct behaviour, not failures.
"""

from __future__ import annotations

import logging
import os
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.config import LLMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a precise question-answering assistant. "
    "Your sole task is to answer questions based ONLY on the context provided below. "
    "Rules you must follow:\n"
    "  1. Do NOT use any knowledge outside the provided context.\n"
    "  2. If the answer is not explicitly present in the context, respond with:\n"
    '     "I cannot answer this question based on the provided context."\n'
    "  3. Be concise and factually accurate.\n"
    "  4. Do not speculate or infer beyond what the context states."
)

_HUMAN = (
    "Context:\n"
    "{context}\n\n"
    "---\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM), ("human", _HUMAN)]
)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def get_llm(config: LLMConfig) -> BaseChatModel:
    """
    Instantiate the LLM specified in *config*.

    Supported providers
    -------------------
    * **openai**    — ``ChatOpenAI``; requires ``OPENAI_API_KEY``.
    * **anthropic** — ``ChatAnthropic``; requires ``ANTHROPIC_API_KEY``.

    Parameters
    ----------
    config:
        LLM provider, model name, and generation hyper-parameters.

    Returns
    -------
    BaseChatModel
        A LangChain chat model ready for ``invoke`` / ``ainvoke``.
    """
    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Export it or add it to your .env file."
            )
        logger.info("Using OpenAI LLM: %s", config.model_name)
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=api_key,
        )

    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it or add it to your .env file."
            )
        logger.info("Using Anthropic LLM: %s", config.model_name)
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            anthropic_api_key=api_key,
        )

    raise ValueError(f"Unsupported LLM provider: '{config.provider}'")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator:
    """
    Grounded answer generator using the RAG prompt template.

    Parameters
    ----------
    config:
        LLM configuration.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.llm = get_llm(config)
        self.prompt = RAG_PROMPT

    def generate(self, question: str, contexts: List[str]) -> str:
        """
        Generate a grounded answer from *question* and *contexts*.

        Contexts are joined with a separator so the model can distinguish
        chunk boundaries without relying on implicit whitespace.

        Parameters
        ----------
        question:
            The natural-language user question.
        contexts:
            Retrieved text chunks from the vector store.

        Returns
        -------
        str
            The generated answer, stripped of leading/trailing whitespace.
        """
        context_block = "\n\n---\n\n".join(contexts)

        messages = self.prompt.format_messages(
            context=context_block,
            question=question,
        )

        response = self.llm.invoke(messages)
        answer: str = response.content.strip()

        logger.debug(
            "Generated %d-char answer for: '%.60s …'", len(answer), question
        )
        return answer

    async def agenerate(self, question: str, contexts: List[str]) -> str:
        """Async variant of :meth:`generate` for use in async pipelines."""
        context_block = "\n\n---\n\n".join(contexts)
        messages = self.prompt.format_messages(
            context=context_block,
            question=question,
        )
        response = await self.llm.ainvoke(messages)
        return response.content.strip()
