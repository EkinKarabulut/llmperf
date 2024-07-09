import argparse
from transformers import AutoTokenizer

def count_tokens(tokenizer_name, text):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def main():
    parser = argparse.ArgumentParser(description="Count the number of tokens in a given text using a specified tokenizer.")
    parser.add_argument('--tokenizer', type=str, required=True, help="The name of the tokenizer to use.")
    parser.add_argument('--text', type=str, required=True, help="The text to tokenize.")

    args = parser.parse_args()

    num_tokens = count_tokens(args.tokenizer, args.text)

    print(f"Number of tokens: {num_tokens}")

if __name__ == "__main__":
    main()

