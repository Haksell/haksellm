import re

import tiktoken

TOKEN_END = "<|endoftext|>"  # from tiktoken


def main():
    with open("the-verdict.txt", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed[:30])

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of someunknownPlace."
    text = f"{text1} {TOKEN_END} {text2}"
    print(text)

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(text, allowed_special={TOKEN_END})
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

    context_size = 4
    enc_sample = enc_text[50:]
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(
            tokenizer.decode(context),
            f"({context})",
            "---->",
            tokenizer.decode([desired]),
            f"({desired})",
        )


if __name__ == "__main__":
    main()
