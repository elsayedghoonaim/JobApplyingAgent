"""Robust extraction of a JSON object from model output."""

import json


def extract_json_object(text: str) -> dict:
    """Decode the first complete JSON object in surrounding model text."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response did not contain a valid JSON object")
