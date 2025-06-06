from enum import Enum, auto
import re
import torch


class Label(Enum):
    CINEMA = auto()
    SCIENCE = auto()
    MUSIC = auto()


LABELS = list(Label)


DATA = [
    ("Movies are fun for everyone.", Label.CINEMA),
    ("Watching movies is great fun.", Label.CINEMA),
    ("Enjoy a great movie today.", Label.CINEMA),
    ("Research is interesting and important.", Label.SCIENCE),
    ("Learning math is very important.", Label.SCIENCE),
    ("Science discovery is interesting.", Label.SCIENCE),
    ("Rock is great to listen to.", Label.MUSIC),
    ("Listen to music for fun.", Label.MUSIC),
    ("Music is fun for everyone.", Label.MUSIC),
    ("Listen to folk music!", Label.MUSIC),
]


def softmax(z):
    e = torch.exp(z)
    return e / e.sum()


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def get_vocabulary(corpus):
    words = sorted({word for document in corpus for word in document})
    return {w: i for i, w in enumerate(words)}


def document_to_bag_of_words(document, vocabulary):
    bag_of_words = [0] * (len(vocabulary) + 1)
    factor = 1 / len(document)
    for word in document:
        bag_of_words[vocabulary.get(word, len(vocabulary))] += factor
    return bag_of_words


def corpus_to_bag_of_words(corpus, vocabulary):
    return torch.tensor(
        [document_to_bag_of_words(document, vocabulary) for document in corpus],
        dtype=torch.float32,
    )


def main():
    corpus = [tokenize(text) for text, _ in DATA]
    vocabulary = get_vocabulary(corpus)
    inputs = corpus_to_bag_of_words(corpus, vocabulary)
    labels = torch.tensor([label.value - 1 for _, label in DATA], dtype=torch.long)

    hidden_nodes = 32
    model = torch.nn.Sequential(
        torch.nn.Linear(len(vocabulary) + 1, hidden_nodes),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_nodes, len(Label)),
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(1 << 10):
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

    new_texts = ["Listening to rock music is fun.", "I love science very much."]
    new_corpus = [tokenize(text) for text in new_texts]
    new_inputs = corpus_to_bag_of_words(new_corpus, vocabulary)
    with torch.no_grad():
        outputs = model(new_inputs)
        predicted_ids = torch.argmax(outputs, dim=1)
    for text, predicted_id in zip(new_texts, predicted_ids):
        print(f"{text} | {LABELS[predicted_id].name}")


if __name__ == "__main__":
    main()
