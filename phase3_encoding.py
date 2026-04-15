# phase3_encoding.py

import numpy as np
import hashlib
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
def generate_watermark_signal(secret_key, length=500):
    # Convert key to hash
    hash_obj = hashlib.sha256(secret_key.encode())
    hash_bytes = hash_obj.digest()

    # Convert to binary
    binary = ''.join(format(byte, '08b') for byte in hash_bytes)

    # Repeat to required length
    binary = (binary * (length // len(binary) + 1))[:length]

    # Convert to signal (-1, +1)
    signal = np.array([1 if b == '1' else -1 for b in binary])
    signal = np.tile(signal, 3)
    return signal
