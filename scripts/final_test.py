import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Define directory structure
base_dir = "D:\\Desktop\\crypto6"
test_dir = os.path.join(base_dir, "data", "test")
model_dir = os.path.join(base_dir, "models", "saved_models")
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Set to run only for round 1
round_num = 2

test_file = os.path.join(test_dir, f"test_dataset_round_{round_num}.csv")
model_file = os.path.join(model_dir, f"model_train_dataset_round_{round_num}_.h5")
output_file = os.path.join(output_dir, f"results_round_{round_num}.csv")

if not os.path.exists(test_file) or not os.path.exists(model_file):
    print(f"Error: Missing test file or model for round {round_num}.")
    exit()

# Load test dataset
test_data = pd.read_csv(test_file)

# Convert hex ciphertext to integer sequences
def hex_to_int_seq(hex_str):
    return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

X_test = [hex_to_int_seq(c) for c in test_data["Ciphertext"]]
X_test_padded = pad_sequences(X_test, maxlen=8, padding='post')

# Load trained model
model = load_model(model_file)

# Predict the plaintext
predictions = model.predict(X_test_padded)
predicted_indices = np.argmax(predictions, axis=-1)
predicted_plaintexts = ["".join([chr(i) for i in row]) for row in predicted_indices]

# Compute character-level accuracy
total_chars = 0
correct_chars = 0
for i in range(len(test_data)):
    true_text = test_data["Plaintext"][i]
    pred_text = predicted_plaintexts[i]
    min_len = min(len(true_text), len(pred_text))
    total_chars += min_len
    correct_chars += sum(1 for j in range(min_len) if true_text[j] == pred_text[j])

character_accuracy = (correct_chars / total_chars) * 100 if total_chars > 0 else 0

# Save results for round 1
df_output = pd.DataFrame({
    'Plaintext': test_data["Plaintext"],
    'Ciphertext': test_data["Ciphertext"],
    'Predicted Plaintext': predicted_plaintexts
})
df_output.to_csv(output_file, index=False)

# Save accuracy summary
summary_file = os.path.join(output_dir, "accuracy_round_1.csv")
pd.DataFrame({"Round": [round_num], "Character-Level Accuracy (%)": [character_accuracy]}).to_csv(summary_file, index=False)

print(f"Results for round {round_num} saved to {output_file}")
print(f"Character-Level Accuracy for round {round_num}: {character_accuracy:.2f}%")
print(f"Accuracy summary saved to {summary_file}")
