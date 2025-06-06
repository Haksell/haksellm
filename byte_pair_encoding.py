from collections import defaultdict
import re


def initialize_vocabulary(corpus):
    vocabulary = defaultdict(int)
    charset = set()
    for word in corpus:
        word_with_marker = "_" + word
        characters = list(word_with_marker)
        charset.update(characters)
        tokenized_word = " ".join(characters)
        vocabulary[tokenized_word] += 1
    return vocabulary, charset


def get_pair_counts(vocabulary):
    pair_counts = defaultdict(int)
    for tokenized_word, count in vocabulary.items():
        tokens = tokenized_word.split()
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += count
    return pair_counts


def merge_pair(vocabulary, pair):
    new_vocabulary = dict()
    pattern = re.compile(r"\b" + re.escape(" ".join(pair)) + r"\b")
    for tokenized_word, count in vocabulary.items():
        new_tokenized_word = re.sub(pattern, "".join(pair), tokenized_word)
        new_vocabulary[new_tokenized_word] = count
    return new_vocabulary


def byte_pair_encoding(corpus, vocab_size):
    vocabulary, charset = initialize_vocabulary(corpus)
    merges = []
    tokens = set(charset)
    while len(tokens) < vocab_size:
        pair_counts = get_pair_counts(vocabulary)
        if not pair_counts:
            break
        most_frequent_pair = max(pair_counts, key=pair_counts.get)
        merges.append(most_frequent_pair)
        vocabulary = merge_pair(vocabulary, most_frequent_pair)
        new_token = "".join(most_frequent_pair)
        tokens.add(new_token)
    return vocabulary, merges, charset, tokens


def tokenize_word(word, merges, vocabulary, charset, unk_token="<UNK>"):
    word = "_" + word
    if word in vocabulary:
        return [word]
    tokens = [c if c in charset else unk_token for c in word]
    for left, right in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i : i + 2] == [left, right]:
                tokens[i : i + 2] = [left + right]
            else:
                i += 1
    return tokens


def text_to_corpus(text):
    return "".join(c.lower() if c.isalnum() else " " for c in text).split()


def main():
    text = open("text/bee_movie.txt").read()
    corpus = text_to_corpus(text)
    vocabulary, merges, charset, tokens = byte_pair_encoding(corpus, 1024)
    new_text = open("text/hooked_on_a_feeling.txt").read()
    new_corpus = text_to_corpus(new_text)
    out_tokens = [
        t for w in new_corpus for t in tokenize_word(w, merges, vocabulary, charset)
    ]
    print(out_tokens)


if __name__ == "__main__":
    main()
