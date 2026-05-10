"""Prompt helpers for grounded Icelandic consumer-rights answers."""

from __future__ import annotations

import os
from typing import Any


SYSTEM_PROMPT = """Þú ert íslenskt spurningasvörunarkerfi um neytendarétt.

Svaraðu aðeins út frá heimildabrotunum sem fylgja með. Ekki nota utanaðkomandi þekkingu. Ef heimildabrotin styðja ekki öruggt svar skaltu segja: "Ég finn ekki nægar upplýsingar í heimildunum til að svara þessu örugglega."

Svaraðu á skýrri íslensku. Hafðu svarið stutt, hagnýtt og varfært. Ekki setja fram lögfræðiráðgjöf sem endanlega niðurstöðu. Vísaðu í heimildir með númerum eins og [1], [2] eða [3]."""


PROMPT_PROFILE_EXTRAS = {
    "balanced": "",
    "strict": (
        "\n\nLeggðu aukna áherslu á að hver fullyrðing sé studd af heimild. "
        "Ef heimildirnar eru óljósar skaltu svara varfærnislega frekar en að fylla í eyður."
    ),
    "user_friendly": (
        "\n\nNotaðu einfalt, notendavænt mál og byrjaðu á beinu svari. "
        "Haltu samt öllum lagalegum fyrirvörum og heimildatilvísunum."
    ),
}


def get_prompt_profile() -> str:
    """Return the selected prompt profile for reportable experiments."""
    profile = os.getenv("PROMPT_PROFILE", "balanced").strip().lower()
    return profile if profile in PROMPT_PROFILE_EXTRAS else "balanced"


def get_system_prompt(profile: str | None = None) -> str:
    """Build the active system prompt from the selected prompt profile."""
    active_profile = profile or get_prompt_profile()
    return SYSTEM_PROMPT + PROMPT_PROFILE_EXTRAS.get(active_profile, "")


def build_answer_prompt(question: str, chunks: list[dict[str, Any]], max_chunks: int = 3) -> str:
    """Build a user prompt from a question and ranked retrieval chunks."""
    selected_chunks = chunks[:max_chunks]

    if not selected_chunks:
        source_text = "Engin heimildabrot fundust."
    else:
        source_text = "\n\n".join(
            _format_chunk(index=index, chunk=chunk)
            for index, chunk in enumerate(selected_chunks, start=1)
        )

    return f"""Spurning notanda:
{question}

Heimildabrot:
{source_text}

Verkefni:
1. Svaraðu spurningunni í 2-5 setningum.
2. Notaðu aðeins upplýsingar sem koma fram í heimildabrotunum.
3. Settu heimildanúmer við mikilvægar fullyrðingar.
4. Ef heimildirnar nægja ekki, segðu það skýrt.
5. Endaðu á stuttri línu: "Heimildir: [x], [y]"
"""


def _format_chunk(index: int, chunk: dict[str, Any]) -> str:
    """Format one retrieved chunk as a numbered source block for the LLM."""
    return f"""[{index}]
Titill: {chunk.get("title", "Óþekktur titill")}
Heimild: {chunk.get("source", "Óþekkt heimild")}
Kafli: {chunk.get("section", "Ótilgreint")}
Slóð: {chunk.get("url", "Ótilgreind slóð")}
Texti: {chunk.get("text", "")}"""
