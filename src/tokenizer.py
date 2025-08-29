from __future__ import annotations

import os
from typing import Any, BinaryIO
from collections.abc import Iterable, Callable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor, nn
from multiprocessing import Pool
import regex as re
from collections import Counter
import json
from einops import rearrange, einsum, repeat
import math
from typing import Union, Optional
import numpy as np


class BPETokenizer:
    def __init__(
        self, 
        pattern: str,
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        # special_tokens: list[str] | None = None
    ):
        ## note that vocab may not start with ascii
        self.pattern = pattern
        self.vocab = vocab
        self.merges = merges
        self.vocab_inv: dict[bytes, int] = {tok: tid for tid, tok in vocab.items()}
        self.id_merges: list[tuple[int, int, int]] = []

        for left, right in merges:
            left_id = self.vocab_inv[left]
            right_id = self.vocab_inv[right]
            self.id_merges.append((left_id, right_id, self.vocab_inv[left + right]))    

        ## no special tokens
        # if special_tokens is not None:
        #     ## sanity check: special tokens must be in vocab
        #     ordered = sorted(special_tokens, key=len, reverse=True)
        #     self.special_tokens = ordered
        #     for special_token in ordered:
        #         if special_token.encode("utf-8") not in self.vocab_inv:
        #             raise ValueError(f"## special_token not in vocab: {special_token}")      
        # else:
        #     self.special_tokens = None

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        ## pre-tokenize the text

        PAT = self.pattern 
        # r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        cnt = Counter()
        for match in re.finditer(PAT, text):
            pretok_str = match.group(0)
            cnt[pretok_str] += 1

        # print(f"## cnt: {cnt}")

        pretok_to_id = {}
        pretoks = []
        for pretok_str, num in cnt.items():
            pretok_to_id[pretok_str] = len(pretoks)
            pretok_bytes = pretok_str.encode("utf-8")
            
            # for b in pretok_bytes:
            #     bb = bytes([b])
            #     if bb == b" ":
            #         raise ValueError(f"## b: {b}, bb: {bb}, self.vocab_inv[bytes([b])]: {self.vocab_inv[bytes([b])]}")
            #     if bb not in self.vocab_inv:
            #         raise ValueError(f"## {b} not in vocab_inv: {bb}")

            pretok = [self.vocab_inv[bytes([b])] for b in pretok_bytes]    
            pretoks.append(PreToken(pretok, num, pretok_str))

        ## -------------------------------------- debug --------------------------------------
        #     if flag:
        #         print(f"## pretok_str: {pretok_str}")
        #         print(f"## pretok: {pretok}")
        #         for b in pretok_bytes:
        #             print(f"## b: {b}, self.vocab_inv[bytes([b])]: {self.vocab_inv[bytes([b])]}")

        # if flag:
        #     print(f"## pretok_to_id: {pretok_to_id}")
        #     print(f"## pretok.tok: {pretoks[0].tok}")
        #     print(f"## cnt: {cnt}")

        # encode the pre-tokens
        for pretok in pretoks:
            for id1, id2, id3 in self.id_merges:
                pretok.only_pop(id1, id2, id3)

        ## merge the encodings
        tokens = []
        for match in re.finditer(PAT, text):
            pretok_id = pretok_to_id[match.group(0)]
            tokens += pretoks[pretok_id].tok

        return tokens

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        ## concatenate self.vocab[id] for id in ids
        bytes_ids = b''.join([self.vocab[id] for id in ids])
        ## decode and handle malformed bytes
        return bytes_ids.decode("utf-8", errors="replace")


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    ## -------------------------------------- debug --------------------------------------
    # print(f"## chunk_size: {chunk_size}")  
    # print(f"## file_size: {file_size}")
    # print(f"## desired_num_chunks: {desired_num_chunks}")
    # print(f"## split_special_token: {split_special_token}")

    # ## test finding boundaries bruteforcelly
    # file.seek(0)
    # large_chunk = file.read(file_size)
    # found_at = large_chunk.find(split_special_token)
    # if found_at != -1:
    #     print(f"## found at: {found_at}")
    # else:
    #     print("## not found")
    ## -------------------------------------- debug --------------------------------------

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def tokenize_chunk(chunk: str, special_tokens: list[str]) -> Counter:
    """
    Use special token pattern to split the chunk first, and the use PAT template 
    to pre-tokenize the sub-chunks.
    """
    split_special_token_str = "|".join(re.escape(tok) for tok in special_tokens)
    split_special_token_pattern = re.compile(split_special_token_str)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    cnt = Counter()
    for subchunk in split_special_token_pattern.split(chunk):
        for match in re.finditer(PAT, subchunk):
            tok = match.group(0).encode("utf-8")
            cnt[tok] += 1

    return cnt


class PreToken:
    def __init__(self, pre_token: bytes, num: int, pre_token_str: str | None = None):
        self.num = num                          ## number of occurrences of this pre-token
        self.tok = [b for b in pre_token]       ## tokenization of this pre-token
        self.pre_token_str = pre_token_str      ## reserved for future use

    def pop(self, id1: int, id2: int, id3: int) -> Counter:
        """
        When id1 and id2 are merged into id3, re-compute the tokenization of 
        this pre-token, and return a counter showing the difference.
        """
        diff_counter = Counter()
        new_tok = []
        i = 0
        while i < len(self.tok):
            if i + 1 < len(self.tok) and self.tok[i] == id1 and self.tok[i + 1] == id2:
                new_tok.append(id3)
                if i > 0:
                    diff_counter[(self.tok[i - 1], id3)] += self.num
                    diff_counter[(self.tok[i - 1], id1)] -= self.num
                if i + 2 < len(self.tok):
                    diff_counter[(id3, self.tok[i + 2])] += self.num
                    diff_counter[(id2, self.tok[i + 2])] -= self.num
                i += 2
            else:
                new_tok.append(self.tok[i])
                i += 1
        self.tok = new_tok
        return diff_counter
    
    def only_pop(self, id1: int, id2: int, id3: int):
        """
        pop but does not return counter
        """
        new_tok = []
        i = 0
        while i < len(self.tok):
            if i + 1 < len(self.tok) and self.tok[i] == id1 and self.tok[i + 1] == id2:
                new_tok.append(id3)
                i += 2
            else:
                new_tok.append(self.tok[i])
                i += 1
        self.tok = new_tok
            

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    ## Parallel pre-tokenization, get a (bytes, cnt) counter
    desired_num_processes = min(16, os.cpu_count())
    chunks = []
    assert "<|endoftext|>" in special_tokens, "<|endoftext|> must be contained in special_tokens"
    print(f"## Want to pre-tokenize with {desired_num_processes} processes")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_processes, "<|endoftext|>".encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    num_processes = len(chunks)
    print(f"## Actually pre-tokenize with {num_processes} processes")
    with Pool(num_processes) as p:
        counters = p.starmap(tokenize_chunk, \
                                    zip(chunks, [special_tokens] * num_processes))
        
    pre_tok_cnt = sum(counters, Counter())
    ## -------------------------------------- debug --------------------------------------
    # print(f"## pre_tok_cnt[:10], {pre_tok_cnt.most_common()[:10]}")

    ## Initialize a pre_token object for all pre-tokens, 
    ## an (int, int) -> int adjacency counter, 
    ## and a dict: (int, int) -> list[int] to record which pre-tokens 
    ## contain that adjacency
    pre_tokens = []
    adj_counter = Counter()
    appear_in : dict[tuple[int, int], list[int]] = dict()
    for tid, (pre_tok_bytes, num) in enumerate(pre_tok_cnt.items()):
        pre_token = PreToken(pre_tok_bytes, num)
        pre_tokens.append(pre_token)
        for i, j in zip(pre_token.tok[:-1], pre_token.tok[1:]):
            adj_counter[(i, j)] += num
            if (i, j) not in appear_in:
                appear_in[(i, j)] = []
            appear_in[(i, j)].append(tid)

    ## -------------------------------------- debug --------------------------------------
    # print("num of pre-tokens: ", tid)
    # for i in range(10):
    #     print(f"## pre_tokens[{i}]: {pre_tokens[i].num}, {pre_tokens[i].tok}")
    # for id, ((i, j), v) in enumerate(adj_counter.most_common()):
    #     print(f"## adj_counter[{i}, {j}]: {v}")
    #     print(f"## appear_in[{i}, {j}]: {appear_in[(i, j)]}")
    #     print(f"## [{i}, {j}] appears in {len(appear_in[(i, j)])} pre-tokens")
    #     if id > 10:
    #         break

    ## BPE merge
    vocab : dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    merges : list[tuple[bytes, bytes]] = []
    cur_vocab = 256
    for special_token in special_tokens:
        vocab[cur_vocab] = special_token.encode('utf-8')
        cur_vocab += 1
    assert cur_vocab <= vocab_size, "current vocab size must not exceed desired vocab_size"
    print(f"## num_merges: {vocab_size - cur_vocab}")

    while cur_vocab < vocab_size:
        ## find the most frequent adjacency, lexicographically largest
        id1, id2 = max(adj_counter, key=lambda x: (adj_counter[x], vocab[x[0]], vocab[x[1]]))

        if cur_vocab % 100 == 0:
            print(f"## cur_vocab: {cur_vocab} / target: {vocab_size}")
        
        ## -------------------------------------- debug --------------------------------------
        # if cur_vocab == 256 + len(special_tokens) + 92:
        #     print(f"################### cur_vocab: {cur_vocab}")
        #     print(f"## max_adj_tokens: {vocab[id1]}, {vocab[id2]}")
        #     for id, ((i, j), v) in enumerate(adj_counter.most_common()):
        #         print(f"## adj_counter[{vocab[i]}, {vocab[j]}]: {v}")
        #         print(f"## appear_in[{i}, {j}]: {appear_in[(i, j)]}")
        #         print(f"## [{i}, {j}] appears in {len(appear_in[(i, j)])} pre-tokens")
        #         if id > 10:
        #             break
                        
        new_token = vocab[id1] + vocab[id2]
        vocab[cur_vocab] = new_token
        merges.append((vocab[id1], vocab[id2]))
        adj_counter.pop((id1, id2))
        appear_in_pre_tokens = appear_in[(id1, id2)]

        for tid in appear_in_pre_tokens:
            diff_counter = pre_tokens[tid].pop(id1, id2, cur_vocab)
            for (i, j), v in diff_counter.items():
                adj_counter[(i, j)] += v
                if v > 0:   
                    if (i, j) not in appear_in:
                        appear_in[(i, j)] = []
                    appear_in[(i, j)].append(tid)

        cur_vocab += 1

    return vocab, merges
