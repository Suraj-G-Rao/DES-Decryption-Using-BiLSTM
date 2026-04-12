import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, TimeDistributed, Dropout, LayerNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define directory structure
base_dir = "D:\\Desktop\\crypto6"
train_dir = os.path.join(base_dir, "data", "train")
model_dir = os.path.join(base_dir, "models", "saved_models")
os.makedirs(model_dir, exist_ok=True)

# Function to convert hex ciphertext to integer sequences
def hex_to_int_seq(hex_str):
    return [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]

# Function to convert plaintext to ASCII integer values
def text_to_indices(text):
    return [ord(c) for c in text]

# Train model for round 1 and round 2
for round_num in range(3, 17):  # Running for rounds 1 and 2
    train_file = os.path.join(train_dir, f"train_dataset_round_{round_num}.csv")
    model_file = os.path.join(model_dir, f"model_train_dataset_round_{round_num}.h5")

    if not os.path.exists(train_file):
        print(f"Skipping round {round_num}: Missing training dataset.")
        continue

    print(f"Training model for round {round_num}...")

    # Load the training dataset
    train_data = pd.read_csv(train_file)

    # Preprocess the data
    X = [hex_to_int_seq(c) for c in train_data["Ciphertext"]]
    y = [text_to_indices(p) for p in train_data["Plaintext"]]

    # Padding sequences
    X_padded = pad_sequences(X, maxlen=8, padding='post')
    y_padded = pad_sequences(y, maxlen=8, padding='post')

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)

    # Define an improved BiLSTM model
    model = Sequential([
        Embedding(input_dim=256, output_dim=256, input_length=8),  # Embedding size maintained
        Bidirectional(LSTM(512, return_sequences=True)),  # Increased LSTM units
        LayerNormalization(),
        Dropout(0.5),  # Stronger dropout to prevent overfitting
        TimeDistributed(Dense(512, activation='relu')),  # Fully connected layer
        TimeDistributed(Dense(128, activation='softmax'))  # Predict one character per timestep
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks for better training
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[early_stopping, lr_scheduler])

    # Save the model
    model.save(model_file)
    print(f"Model for round {round_num} trained and saved successfully.")
