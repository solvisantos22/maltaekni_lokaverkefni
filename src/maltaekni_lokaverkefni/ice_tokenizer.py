"""Icelandic tokenization and lemmatization helpers."""

from __future__ import annotations

import re

try:
    from tokenizer import tokenize
except ImportError:  # Allows lexical retrieval to run without the tokenizer package.
    tokenize = None

try:
    from reynir import Greynir
except ImportError:  # Reynir is preferred, but Windows/demo environments may lack it.
    Greynir = None


class IceTokenizer:
    """
    Icelandic tokenization and lemmatization wrapper.

    Explanation:
        IceTokenizer provides a small project-specific interface around the
        tokenizer and Reynir libraries. When Reynir is installed, it lemmatizes
        Icelandic text so lexical retrieval can compare normalized word forms
        instead of only exact surface forms. When Reynir is unavailable, it
        falls back to lowercase regex tokens so the app still starts and can be
        demonstrated locally. The original text is never changed for display;
        these normalized tokens are only used for indexing and matching.

    Attributes:
        greynir: Reynir parser instance used for Icelandic morphological
            analysis and lemmatization.

    Public methods:
        tokenIce(text): Tokenize one string or a list of strings.
        lemmatIce(text): Lemmatize one string or a list of strings.
    """

    def __init__(self):
        """Create the Reynir parser when available, otherwise use fallback tokens."""
        self.greynir = Greynir() if Greynir is not None else None
        self.backend = "reynir" if self.greynir is not None else "regex"

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
        if tokenize is None:
            return self.__regex_tokens(text)

        return [
            token_text
            for token in tokenize(text)
            if (token_text := getattr(token, "txt", "").strip().lower())
            and re.search(r"\w", token_text)
        ]

    def __lemmatize(self, text: str) -> list[str]:
        """Return lemmas for incoming Icelandic text.

        Reynir exposes sentence-level lemmas when parsing succeeds. If a
        sentence has no lemma analysis, the tokenizer falls back to lowercased
        token text so retrieval can still index the source instead of dropping
        it completely.
        """
        if self.greynir is None:
            return self.__regex_tokens(text)

        lemmas = []
        try:
            job = self.greynir.parse(text)
        except Exception:
            return self.__regex_tokens(text)

        for sentence in job["sentences"]:
            if not sentence.lemmas:
                for token in sentence.tokens:
                    token_text = getattr(token, "txt", "").strip().lower()
                    if token_text:
                        lemmas.append(token_text)
            else:
                for lemma in sentence.lemmas:
                    if lemma is not None:
                        lemmas.append(lemma.lower())
        
        return [lemma for lemma in lemmas if (re.search(r"\w", lemma))]

    def __regex_tokens(self, text: str) -> list[str]:
        """Fallback analyzer used when Icelandic NLP packages are unavailable."""
        return re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)



if __name__ == "__main__":
    it = IceTokenizer()
    print(it.lemmatIce("Hvað get ég gert ef vara sem ég keypti er gölluð?"))
