import os
import pytest
from transformers import AutoTokenizer
from .common import FIXTURES_PATH

from .qwen_tokenizer import (
    encode_file_with_qwen_tokenizer,
    decode_sequence_with_qwen_tokenizer,
)


@pytest.fixture(scope="module")
def setup_test_environment():
    """
    Creates a temporary directory and a sample input file for testing.
    This fixture is run once per module.
    """
    input_file_path = FIXTURES_PATH / "input.txt"
    sample_text = "Hello world! This is the Qwen tokenizer."

    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
    return {"path": input_file_path, "text": sample_text}


def test_encode_decode_roundtrip(setup_test_environment):
    """
    Tests the full encode -> decode pipeline.
    Ensures that decoding the encoded tokens restores the original text exactly.
    """
    input_path = setup_test_environment["path"]
    original_text = setup_test_environment["text"]

    encoded_ids = encode_file_with_qwen_tokenizer(input_path)
    
    assert isinstance(encoded_ids, list), "encode function should return a list."
    assert all(isinstance(i, int) for i in encoded_ids), "All items in the encoded list should be integers."

    decoded_text = decode_sequence_with_qwen_tokenizer(encoded_ids)

    assert decoded_text == original_text, "Decoded text does not match the original text."


# @pytest.mark.parametrize("model_name", ["Qwen/Qwen1.5-0.5B"])
# def test_encoding_matches_huggingface(setup_test_environment, model_name):
#     """
#     Compares the student's tokenizer output with the official Hugging Face tokenizer.
#     This is a critical test for correctness.
    
#     NOTE: This test requires the `transformers` and `torch` libraries.
#           `pip install transformers torch`
#     """
#     input_path = setup_test_environment["path"]
#     sample_text = setup_test_environment["text"]
    
#     # 1. Encode the text using the official Hugging Face tokenizer
#     try:
#         hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     except Exception as e:
#         pytest.fail(f"Failed to load Hugging Face tokenizer '{model_name}'. Ensure you are online. Error: {e}")
        
#     hf_encoded_ids = hf_tokenizer.encode(sample_text)

#     # 2. Encode the same text using the student's implementation
#     student_encoded_ids = encode_file_with_qwen_tokenizer(input_path)

#     # 3. Assert that the two lists of token IDs are identical
#     assert student_encoded_ids == hf_encoded_ids, \
#         f"Student's token IDs do not match the official Hugging Face tokenizer's IDs." \
#         f"\nStudent:  {student_encoded_ids}" \
#         f"\nOfficial: {hf_encoded_ids}"

