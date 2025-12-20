import re

def preprocess_text(input_file, output_file):
    """
    Preprocess a text file for Word2Vec training.
    
    Steps:
    1. Lowercase all text
    2. Remove punctuation
    3. Replace newlines with spaces
    4. Tokenize words separated by spaces
    5. Write cleaned words to output file (space separated)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation (keep only letters and numbers)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing spaces
    text = text.strip()
    
    # Save preprocessed corpus
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Preprocessing done. Output written to {output_file}")


if __name__ == "__main__":
    input_file = "corpus/alice.txt"  # your raw corpus
    output_file = "corpus/alice_corpus_preped.txt"        # preprocessed corpus
    preprocess_text(input_file, output_file)
