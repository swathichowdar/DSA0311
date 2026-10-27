"""
Experiment 1: Word Tokenization and Sentence Segmentation
Course: Natural Language Processing (NLP)
"""

import re

def sentence_segmentation_regex(text: str) -> list[str]:
    """
    Segments a paragraph into sentences using regular expressions.
    Handles standard terminal punctuation: '.', '!', and '?'.
    """
    pattern = r'(?<=[.!?])\s+'
    sentences = [s.strip() for s in re.split(pattern, text) if s.strip()]
    return sentences

def word_tokenization_regex(sentence: str) -> list[str]:
    """
    Tokenizes a sentence into words and punctuation marks using regex.
    Preserves words, contractions, and individual punctuation marks.
    """
    pattern = r"\b\w+(?:'\w+)?\b|[^\w\s]"
    tokens = re.findall(pattern, sentence)
    return tokens

def whitespace_tokenization(text: str) -> list[str]:
    """Splits text on whitespace boundaries."""
    return text.split()

def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 1: WORD TOKENIZATION AND SENTENCE SEGMENTATION")
    print("=" * 70)

    sample_corpus = (
        "Natural Language Processing (NLP) is a subfield of Artificial Intelligence (AI). "
        "It enables computers to understand, process, and generate human language. "
        "Can machines truly comprehend human emotions? Yes, through sentiment analysis! "
        "Don't worry, mastering NLP concepts step-by-step is both practical and exciting."
    )

    print("\n[INPUT CORPUS]")
    print(sample_corpus)
    print("-" * 70)

    # 1. Sentence Segmentation
    print("\n--- 1. Sentence Segmentation (Regex-Based) ---")
    sentences = sentence_segmentation_regex(sample_corpus)
    for idx, sentence in enumerate(sentences, start=1):
        print(f"Sentence {idx}: {sentence}")

    # 2. Word Tokenization Comparison
    print("\n--- 2. Word Tokenization on Sentence 1 ---")
    s1 = sentences[0]
    print(f"Target Sentence: \"{s1}\"")

    whitespace_tokens = whitespace_tokenization(s1)
    regex_tokens = word_tokenization_regex(s1)

    print(f"\nWhitespace Tokenization ({len(whitespace_tokens)} tokens):")
    print(whitespace_tokens)

    print(f"\nRegex / Linguistic Tokenization ({len(regex_tokens)} tokens):")
    print(regex_tokens)

    # 3. Full Corpus Tokenization Breakdown
    print("\n--- 3. Full Corpus Tokenization Breakdown ---")
    all_tokens = []
    for idx, s in enumerate(sentences, start=1):
        tokens = word_tokenization_regex(s)
        all_tokens.extend(tokens)
        print(f"Sentence {idx} ({len(tokens)} tokens): {tokens}")

    print("\n" + "-" * 70)
    print(f"Summary Statistics:")
    print(f" - Total Sentences: {len(sentences)}")
    print(f" - Total Word/Punctuation Tokens: {len(all_tokens)}")
    print(f" - Unique Vocabulary Count: {len(set(t.lower() for t in all_tokens))}")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
