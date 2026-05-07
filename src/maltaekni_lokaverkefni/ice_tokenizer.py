"""Icelandic tokenization and lemmatization helpers."""

from __future__ import annotations

import re

from tokenizer import tokenize

try:
    from reynir import Greynir
except ImportError:
    Greynir = None


class IceTokenizer:
    """
    Icelandic tokenization and lemmatization wrapper.

    Explanation:
        IceTokenizer provides a small project-specific interface around the
        tokenizer and Reynir libraries. It can tokenize raw Icelandic text or
        lemmatize it so TF-IDF retrieval can compare normalized word forms
        instead of only exact surface forms.

    Attributes:
        greynir: Reynir parser instance used for Icelandic morphological
            analysis and lemmatization.

    Public methods:
        tokenIce(text): Tokenize one string or a list of strings.
        lemmatIce(text): Lemmatize one string or a list of strings.
    """

    def __init__(self):
        """Create the Reynir parser used by this tokenizer."""
        self.greynir = Greynir() if Greynir is not None else None

    def tokenIce(self, text: str | list[str]):
        """Tokenize a string or list of strings into Icelandic tokens."""
        if isinstance(text, list):
            return [self.__tokenize(item) for item in text]

        return self.__tokenize(text)

    def lemmatIce(self, text: str | list[str]):
        """Lemmatize a string or list of strings."""
        if isinstance(text, list):
            return [self.__lemmatize(item) for item in text]

        return self.__lemmatize(text)

    def __tokenize(self, text: str) -> list[str]:
        """Tokenize one string and return token text values."""
        return [
            token_text
            for token in tokenize(text)
            if (token_text := getattr(token, "txt", "").strip().lower())
            and re.search(r"\w", token_text)
        ]

    def __lemmatize(self, text: str) -> list[str]:
        """Return lemmas for incoming Icelandic text.

        The original text should still be kept for display and citations. Use
        this output only for searchable/indexed text.

        Reynir token analyses expose candidate meanings in token.val. The first
        item in the first analysis tuple is the lemma, for example "Mig" maps
        to "ég".
        """
        if self.greynir is None:
            return self.__tokenize(text)

        lemmas = []
        job = self.greynir.parse(text)
        for sentence in job["sentences"]:
            for token in sentence.tokens:
                lemma = token.val[0][0]
                if lemma is not None:
                    lemmas.append(lemma.lower())
        
        return lemmas



if __name__ == "__main__":
    it = IceTokenizer()
    print(it.lemmatIce("Mig þig Jóhannesi ferðaðist rauðasti bíllinn"))
