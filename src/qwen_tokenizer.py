import os
import json
from .tokenizer import BPETokenizer
from .common import FIXTURES_PATH

## given 

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def load_vocab_and_merges(
    directory: str = FIXTURES_PATH / 'qwen3_tokenizer',
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab_path = os.path.join(directory, 'vocab.json')
    merges_path = os.path.join(directory, 'merges.txt')
    
    ## load actual vocab and merge files, return vocab and merges after processing (bytes to unicode)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        raw_vocab = json.load(f)
    vocab = {v: k.encode('utf-8') for k, v in raw_vocab.items()}

    with open(merges_path, 'r', encoding='utf-8') as f:
        merge_lines = f.read().split('\n')[1:-1]
    merges = [
        (s1.encode('utf-8'), s2.encode('utf-8'))
        for merge_str in merge_lines
        for s1, s2 in [tuple(merge_str.split())]
    ]

    return vocab, merges


def load_tokenizer_from_dir(
    # directory: str = FIXTURES_PATH / 'qwen3_tokenizer',
    # pattern: str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
) -> BPETokenizer:
    """
    Loads tokenizer files from a directory and instantiates a BPETokenizer.

    Args:
        directory (str): The path to the directory containing 'vocab.json',
                         'merges.txt', and 'tokenizer_config.json'.

    Returns:
        An instantiated BPETokenizer.

    Do not use AutoTokenizer.
    """

    ## write bytes to unicode here
    ## 

    vocab = ...
    merges = ...


    # raise NotImplementedError

    vocab_path = os.path.join(directory, 'vocab.json')
    merges_path = os.path.join(directory, 'merges.txt')

    with open(vocab_path, 'r', encoding='utf-8') as f:
        raw_vocab = json.load(f)
    vocab = {v: k.encode('utf-8') for k, v in raw_vocab.items()}

    with open(merges_path, 'r', encoding='utf-8') as f:
        merge_lines = f.read().split('\n')[1:-1]
    merges = [
        (s1.encode('utf-8'), s2.encode('utf-8'))
        for merge_str in merge_lines
        for s1, s2 in [tuple(merge_str.split())]
    ]
    
    return BPETokenizer(pattern=pattern, vocab=vocab, merges=merges)


def encode_file_with_qwen_tokenizer(
    input_path: str,
) -> list[int]:
    """
    Reads text from an input file, encodes it using the qwen tokenizer, and 
    returns the list of token IDs.
    """
    tokenizer = create_tokenizer_from_dir()

    # raise NotImplementedError
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()        
    return tokenizer.encode(text)


def decode_file_with_qwen_tokenizer(
    token_ids: list[int],
) -> str:
    """
    Decodes a list of token IDs using the qwen tokenizer and returns the decoded text.
    """
    tokenizer = create_tokenizer_from_dir()

    # raise NotImplementedError

    return tokenizer.decode(token_ids)
