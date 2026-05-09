from __future__ import annotations

from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
from dataclasses import asdict
from pathlib import Path
import re

try:
    from .types_classes import Document
except ImportError:  # Allows direct script execution during early experiments.
    from types_classes import Document



class FetchData:
    """
    Fetch and parse one curated source page into a Document.

    Explanation:
        This class downloads one HTML page from an approved source and parses it
        into clean text that can later be chunked for retrieval. At the moment
        the implemented parser targets Althingi law pages from the Lagasafn.

    Attributes:
        url: URL of the page being fetched.
        source: Normalized source name, currently "althingi" or
            "neytendastofa".
        timeout_seconds: Network timeout used when fetching the page.
        parsed: Clean text extracted from the downloaded HTML.
        html: Raw downloaded HTML.
        title: Title inferred from parsed text.
        doc_id: Stable document identifier inferred from the URL.

    Public methods:
        fetch(url, timeout_seconds=60): Download and store raw HTML for a URL.
        data(): Parse the downloaded HTML and return a Document object.
    """
    
    url: str | None = None
    source: str| None = None
    timeout_seconds: int| None = None
    parsed: str| None = None
    html: str| None = None
    title: str | None = None
    doc_id: str | None = None
    
    def fetch(self, url: str, timeout_seconds: int = 60):
        """Download a supported source URL and store the raw HTML."""
        if 'althingi' in url:
            self.source = 'althingi'
        else:
            raise ValueError(f"only urls from althingi supported, got {url}")
        
        self.url = url
        self.timeout_seconds = timeout_seconds
    
        self.html = self.__download_html()
        


    def __download_html(self) -> str:
        """Download HTML for self.url and decode it to text."""
        request = Request(
            self.url,
            headers={
                "User-Agent": "maltaekni-lokaverkefni/0.1 educational RAG project",
                "Accept": "text/html,application/xhtml+xml",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                content_type = response.headers.get("content-type", "")
                if "html" not in content_type.lower():
                    raise ValueError(f"Expected HTML from {self.url}, got {content_type!r}")

                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except HTTPError as error:
            raise RuntimeError(f"HTTP {error.code} while fetching {self.url}") from error
        except URLError as error:
            raise RuntimeError(f"Could not fetch {self.url}: {error.reason}") from error


    def __parse(self):
        """
        Parse self.html with the parser that matches self.source.
        """
        
        if self.source == 'althingi':
            parser = _AlthingiParser()
        else:
            raise NotImplementedError("Only Althingi parsing is implemented for now")
        
        parser.feed(self.html)

        parsed = "\n".join(
            line for line in (
                " ".join(part.replace("\xa0", " ").split())
                for part in parser.parts
            )
            if line
        )

        if not parsed:
            raise ValueError("Could not parse Althingi law text from html")
        else:
            self.parsed = parsed
            
    
    def __get_title_from_parsed(self) -> str:
        """Infer a document title from the parsed text."""
        lines = [line.strip() for line in self.parsed.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("Lög um "):
                self.title = line

        self.title = lines[0] if lines else "Óþekktur titill"
    
    def __get_document_id_from_url(self) -> int:
        """Infer a stable document identifier from the page URL."""
        matches = re.findall(r'[0-9]+\.html', self.url)
        self.doc_id = matches[0] 
    
    def data(self):
        """Return the fetched page as a parsed Document object."""
        if not self.parsed:
           self.__parse()
        if not self.title:
           self.__get_title_from_parsed()
        if not self.doc_id:
           self.__get_document_id_from_url()
        
        return Document(
            url = self.url,
            source = self.source,
            title = self.title,
            text = self.parsed,
            document_id = self.doc_id
        )
          

class _AlthingiParser(HTMLParser):
    """
    Extract law text from Althingi law pages.

    Explanation:
        Althingi Lagasafn pages contain the useful law text inside
        <div class="article box login"> ... </div>. This parser ignores page
        navigation and metadata, keeps article headings and list items readable,
        and skips amendment-note artifacts such as superscript footnotes and
        small-print amendment references.

    Attributes:
        parts: Completed output lines extracted from the law body.

    Public methods:
        feed(html): Inherited from HTMLParser. Parse an HTML string and fill
            parts with cleaned law-text lines.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.__line = []
        self.__inside_law = False
        self.__div_depth = 0
        self.__skip_depth = 0

    def handle_starttag(self, tag, attrs):
        """Handle opening HTML tags and detect the law text container."""
        attrs = dict(attrs)

        if tag == "div" and self.__is_law_container(attrs):
            self.__inside_law = True
            self.__div_depth = 1
            return

        if not self.__inside_law:
            return

        if tag == "div":
            self.__div_depth += 1

        if tag in {"script", "style", "noscript", "svg", "sup", "small"}:
            self.__skip_depth += 1
            return

        if tag == "img" and attrs.get("src") in {"/lagas/hk.jpg", "/lagas/sk.jpg"}:
            self.__new_line()
            return

        if tag in {"br", "p", "h1", "h2", "h3", "h4", "hr"}:
            self.__new_line()

    def handle_endtag(self, tag):
        """Handle closing HTML tags and flush complete text lines."""
        if not self.__inside_law:
            return

        if tag in {"script", "style", "noscript", "svg", "sup", "small"} and self.__skip_depth:
            self.__skip_depth -= 1

        if tag in {"br", "p", "h1", "h2", "h3", "h4", "hr"}:
            self.__new_line()

        if tag == "div":
            self.__div_depth -= 1
            if self.__div_depth <= 0:
                self.__new_line()
                self.__inside_law = False

    def handle_data(self, data):
        """Collect readable text from inside the law text container."""
        if not self.__inside_law or self.__skip_depth:
            return

        text = " ".join(data.replace("\xa0", " ").split())
        if text:
            self.__line.append(text)

    def __is_law_container(self, attrs):
        classes = set(attrs.get("class", "").split())
        return {"article", "box", "login"}.issubset(classes)

    def __new_line(self):
        line = " ".join(self.__line).strip()
        if line:
            self.parts.append(line)
        self.__line = []


if __name__ == "__main__":
    sources = ["https://www.althingi.is/lagas/nuna/2003048.html",
               "https://www.althingi.is/lagas/nuna/2016016.html",
               "https://www.althingi.is/lagas/nuna/2011076.html",
               "https://www.althingi.is/lagas/nuna/2013120.html",
               "https://www.althingi.is/lagas/nuna/2002030.html",
               "https://www.althingi.is/lagas/nuna/2018095.html",
               "https://www.althingi.is/lagas/nuna/2016118.html",
               "https://www.althingi.is/lagas/nuna/2013033.html"]
    
    documents = []
    
    for source in sources:
        f = FetchData()
        f.fetch(source)
        documents.append(asdict(f.data()))
    
    output_path = Path("data/processed/documents.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(documents, file, ensure_ascii=False, indent=2)


fetch_data = FetchData
