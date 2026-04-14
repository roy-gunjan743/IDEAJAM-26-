# phase3_encoding.py

import numpy as np

# Convert text to binary
def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

# Add redundancy (simple ECC)
def add_redundancy(binary):
    return ''.join([bit * 3 for bit in binary])  # repeat 3 times

# Convert to signal (-1, +1)
def binary_to_signal(binary):
    arr = np.array([int(b) for b in binary])
    return np.where(arr == 0, -1, 1)

# MAIN FUNCTION
def generate_watermark_signal(text):
    binary = text_to_binary(text)
    encoded = add_redundancy(binary)
    signal = binary_to_signal(encoded)

    return signal