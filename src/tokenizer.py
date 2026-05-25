"""Character-level tokenizer utilities"""

class CharacterTokenizer:
    """Simple character-level tokenizer"""

    def __init__(self, vocab_file):
        """Load vocabulary from file"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            chars = []
            for line in f:
                if line == '\n':
                    # Empty line represents the newline character itself
                    chars.append('\n')
                else:
                    # Remove trailing newline, keep the character
                    chars.append(line[:-1] if line.endswith('\n') else line)

        self.chars = chars
        self.vocab_size = len(chars)
        self.string_to_int = {ch: i for i, ch in enumerate(chars)}
        self.int_to_string = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, text):
        """Encode text to list of integers"""
        return [self.string_to_int[c] for c in text]
    
    def decode(self, indices):
        """Decode list of integers to text"""
        return ''.join([self.int_to_string[i] for i in indices])
    
    @staticmethod
    def create_vocab(text_file, vocab_file):
        """Create vocabulary file from text file"""
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))

        with open(vocab_file, 'w', encoding='utf-8') as f:
            for char in chars:
                f.write(char + '\n')
            
        return len(chars)
