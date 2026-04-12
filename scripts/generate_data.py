import os
import re
from Crypto.Cipher import DES
import pandas as pd
from PyPDF2 import PdfReader

# Define directory structure
base_dir = "D:\\Desktop\\crypto6"
train_dir = os.path.join(base_dir, "data", "train")
test_dir = os.path.join(base_dir, "data", "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to pad meaningful plaintext to 8 bytes
def pad_plaintext(plaintext):
    plaintext = plaintext[:8]  # Truncate if longer than 8 bytes
    return plaintext.ljust(8, " ")  # Pad with spaces if shorter

# Function to simulate DES encryption for specific rounds
def des_encrypt_with_rounds(key, plaintext, rounds):
    cipher = DES.new(key, DES.MODE_ECB)
    L, R = plaintext[:4].encode('utf-8'), plaintext[4:].encode('utf-8')
    
    for _ in range(rounds):
        expanded_R = cipher.encrypt(R.ljust(8, b'\x00'))  # Encrypt padded R
        new_L = bytes(x ^ y for x, y in zip(L, expanded_R[:4]))
        L, R = R, new_L  # Swap L and R
    
    return (L + R).hex()

# Extract and clean text from PDFs
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    return re.sub(r'[^A-Za-z0-9 ]', '', text.replace("\n", " ").replace("\r", " "))

# Extract plaintexts
train_text = extract_text(os.path.join(base_dir, "Attention.pdf"))
test_text = extract_text(os.path.join(base_dir, "LLM.pdf"))

# Prepare meaningful plaintext phrases
train_blocks = [pad_plaintext(train_text[i:i+8]) for i in range(0, len(train_text), 8)]
test_blocks = [pad_plaintext(test_text[i:i+8]) for i in range(0, len(test_text), 8)]

# Special 36 rows (8-char repeating patterns)
special_plaintexts = [chr(i) * 8 for i in range(97, 123)] + [str(i) * 8 for i in range(10)]  # 'aaaa...' to 'zzzz...' and '0000...' to '9999...'

# Ensure key remains the same across all rounds
key = os.urandom(8)

# Generate datasets for 16 rounds
pairs_per_round_train = 100000
pairs_per_round_test = 20000

for rounds in range(1, 17):
    # Create training dataset with consistent plaintexts
    train_plaintexts = special_plaintexts + train_blocks[:pairs_per_round_train - len(special_plaintexts)]
    train_ciphertexts = [des_encrypt_with_rounds(key, pt, rounds) for pt in train_plaintexts]

    train_file = os.path.join(train_dir, f"train_dataset_round_{rounds}.csv")
    pd.DataFrame({"Plaintext": train_plaintexts, "Ciphertext": train_ciphertexts}).to_csv(train_file, index=False, encoding="utf-8")
    print(f"Saved training dataset for {rounds} rounds: {train_file}")

    # Create testing dataset with consistent plaintexts
    test_plaintexts = special_plaintexts + test_blocks[:pairs_per_round_test - len(special_plaintexts)]
    test_ciphertexts = [des_encrypt_with_rounds(key, pt, rounds) for pt in test_plaintexts]

    test_file = os.path.join(test_dir, f"test_dataset_round_{rounds}.csv")
    pd.DataFrame({"Plaintext": test_plaintexts, "Ciphertext": test_ciphertexts}).to_csv(test_file, index=False, encoding="utf-8")
    print(f"Saved testing dataset for {rounds} rounds: {test_file}")
