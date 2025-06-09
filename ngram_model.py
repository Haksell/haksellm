from collections import defaultdict
import math
from models import load_model, save_model
from prepare import tokenize, train_test_split
import random


# TODO: implement with a trie instead
class CountBasedLanguageModel:
    def __init__(self, n, train_corpus=None):
        self.n = n
        self.ngram_counts = [dict() for _ in range(n)]
        self.total_unigrams = 0
        if train_corpus is not None:
            self.train(train_corpus)

    def train(self, tokens):
        self.total_unigrams = len(tokens)
        for context_len in range(self.n):
            counts = self.ngram_counts[context_len]
            for i in range(len(tokens) - context_len):
                context = tuple(tokens[i : i + context_len])
                next_token = tokens[i + context_len]
                if context not in counts:
                    counts[context] = defaultdict(int)
                counts[context][next_token] += 1

    def predict_next_token(self, context):
        for n in range(min(self.n, len(context) + 1), 0, -1):
            context_n = tuple(context[len(context) - n + 1 :])
            counts = self.ngram_counts[n - 1].get(context_n)
            if counts:
                return max(counts, key=counts.get)

    def get_probability(self, token, context):
        for n in range(min(self.n, len(context) + 1), 1, -1):
            context_n = tuple(context[len(context) - n + 1 :])
            counts = self.ngram_counts[n - 1].get(context_n)
            if counts:
                total = sum(counts.values())
                count = counts.get(token, 0)
                if count > 0:
                    return count / total
        unigram_counts = self.ngram_counts[0].get(())
        count = unigram_counts.get(token, 0)
        return (count + 1) / (self.total_unigrams + len(unigram_counts))

    # Lower perplexity indicates better model performance
    def compute_perplexity(self, tokens):
        if not tokens:
            return math.inf

        total_log_likelihood = sum(
            math.log(self.get_probability(token, tuple(tokens[max(0, i - self.n) : i])))
            for i, token in enumerate(tokens)
        )
        avg_log_likelihood = total_log_likelihood / len(tokens)
        return math.exp(-avg_log_likelihood)


MODEL_NAME = "ngram"
CONTEXT_SIZE = 3


def main():
    random.seed(42)

    # corpus = download_online_corpus("https://www.thelmbook.com/data/brown")
    corpus = open("text/bee_movie.txt").read()
    tokens = tokenize(corpus)
    train_corpus, test_corpus = train_test_split(tokens)
    if (model := load_model(MODEL_NAME)) is None:
        model = CountBasedLanguageModel(CONTEXT_SIZE, train_corpus)
        save_model(model, MODEL_NAME)

    perplexity = model.compute_perplexity(test_corpus)
    print(f"Perplexity on test corpus: {perplexity:.2f}")
    contexts = ["i will build a", "the best place to", "she was riding a"]
    for context in contexts:
        words = tokenize(context)
        next_word = model.predict_next_token(words)
        print(f"{context} -> {next_word}")


if __name__ == "__main__":
    main()
