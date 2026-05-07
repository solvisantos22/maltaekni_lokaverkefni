try:
    from .fetch_sources import FetchData
except ImportError:  # Allows direct script execution during early experiments.
    from fetch_sources import FetchData


def main():
    """Small manual smoke test for fetching one Althingi source."""
    fetcher = FetchData()
    fetcher.fetch("https://www.althingi.is/lagas/nuna/2016016.html")
    print(fetcher.data())


if __name__ == "__main__":
    main()
