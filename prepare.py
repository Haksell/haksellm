import gzip
import io
import re
import requests


def download_online_corpus(url, verbose=False):
    if verbose:
        print(f"Downloading corpus from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    if verbose:
        print("Decompressing and reading the corpus...")
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
        corpus = f.read().decode("utf-8")

    if verbose:
        print(f"Corpus size: {len(corpus)} characters")
    return corpus


def tokenize(text):
    return re.findall(r"\b[a-z0-9]+\b|[.]", text.lower())


def train_test_split(tokens, test_size=0.1):
    split_index = int(len(tokens) * (1 - test_size))
    return tokens[:split_index], tokens[split_index:]
