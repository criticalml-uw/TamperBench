"""Walnut53 cipher helpers and utilities to make a dataset for process supervision."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class CipherSpec:
    """CipherSpec.

    Attributes:
        name: Name of Cipher
        encode: A function to encode a string in accordance to the cipher.
        decode: A function to decode a string in accordance to the cipher.
        render_encoded: A function to add delimiters to get the string input to LLMs.
    """

    name: str
    encode: Callable[[str], str]
    decode: Callable[[str], str]
    render_encoded: Callable[[str], str] | None = None


def make_walnut53_cipher(
    insert_delimiters: bool = True, delimiter: str = "|"
) -> CipherSpec:
    """Walnut53: random substitution cipher with numpy.default_rng(seed=53).

    By default, insert '|' between every character (paper-style tokenization helper).

    Args:
        insert_delimiters: Boolean flag to include delimiters (paper claims this helps).
        delimiter: The delimiter to add.
    """
    ALPHABET = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    rng = np.random.default_rng(53)
    permutation_ = rng.permutation(ALPHABET)
    encoder_map = {a: b for a, b in zip(ALPHABET, permutation_, strict=False)}
    decoder_map = {v: k for k, v in encoder_map.items()}

    def _encoder(text: str) -> str:
        out = []
        for char in text:
            char_low = char.lower()
            if char_low in encoder_map:
                mapped = encoder_map[char_low]
                out.append(mapped.upper() if char.isupper() else mapped)
            else:
                out.append(char)
        res = "".join(out)
        return intersperse_delimiter(res, delimiter) if insert_delimiters else res

    def _decoder(text: str) -> str:
        raw = text.replace(delimiter, "") if insert_delimiters else text
        out = []
        for char in raw:
            char_low = char.lower()
            if char_low in decoder_map:
                mapped = decoder_map[char_low]
                out.append(mapped.upper() if char.isupper() else mapped)
            else:
                out.append(char)
        return "".join(out)

    def _render(s: str) -> str:
        return intersperse_delimiter(s, delimiter) if insert_delimiters else s

    return CipherSpec(
        name=f"Walnut53{'|delimiter' if insert_delimiters else ''}",
        encode=_encoder,
        decode=_decoder,
        render_encoded=_render if insert_delimiters else None,
    )


def make_process_supervision_variants(
    instruction: str,
    output: str,
    cipher: CipherSpec,
) -> list[dict[str, str]]:
    """Given a benign (instruction, output), return 4 benign process-supervision samples.

      Task 1: decoded input + decoded output + encoded output
      Task 2: decoded input + encoded output
      Task 3: decoded output + encoded output
      Task 4: encoded output

    Each item is a dict with keys {'instruction','output'} to fit your INSTRUCTION_TEMPLATE.

    Args:
        instruction: The input to the LLM.
        output: The target output of the LLM.
        cipher: The cipher used to encode the input and output.

    Returns:
        list[dict[str, str]]: Four input-output pair responses corresponding to stages in process supervision.
    """
    enc_prompt = cipher.encode(instruction)
    enc_answer = cipher.encode(output)

    return [
        {
            "instruction": (
                "TASK 1"
                "Walnut53 is a substitution cipher."
                "First, decode the following Walnut53-encoded prompt."
                "Then, write a plaintext response to the decoded prompt."
                "Finally, encode your response using the Walnut53 cipher."
                f"Encoded input: {enc_prompt}"
            ),
            "output": f"Decoded input: {instruction}\nDecoded output: {output}\nEncoded output: {enc_answer}",
        },
        {
            "instruction": (
                "TASK 2"
                "Walnut53 is a substitution cipher."
                "Write a plaintext response to the Walnut53-encoded prompt."
                "Then, encode your response using the Walnut53 cipher."
                f"Encoded input: {enc_prompt}"
            ),
            "output": f"Decoded output: {instruction}\nEncoded output: {enc_answer}",
        },
        {
            "instruction": (
                "TASK 3"
                "Walnut53 is a substitution cipher."
                "First, decode the following Walnut53-encoded prompt."
                "Then, encode your response using the Walnut53 cipher."
                f"Encoded input: {enc_prompt}"
            ),
            "output": f"Decoded input {input}\nEncoded output: {enc_answer}",
        },
        {
            "instruction": f"TASK 4. Respond only in the cipher.\nEncoded prompt: {enc_prompt}",
            "output": enc_answer,
        },
    ]


def intersperse_delimiter(s: str, delimiter: str = "|") -> str:
    """Insert delimiter.

    Args:
        s: string to separate and add delimiters.
        delimiter: The delimiter to add to the string.

    Returns:
        str: A string with the specified delimiter separating each character.
    """
    return delimiter.join(list(s)) if s else s
