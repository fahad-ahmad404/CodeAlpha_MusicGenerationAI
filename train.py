# ==============================================================
# CodeAlpha Artificial Intelligence Internship
# Project      : Music Generation with AI
# Developed By : Fahad Ahmad
# Student ID   : CA/DF1/189817
# Description  : Stage 2 of 3 - Model Training.
#                Loads the prepared note sequences, builds a deep LSTM
#                network using TensorFlow/Keras, and trains it to
#                predict the next note in a sequence. Saves the best
#                model weights during training.
# ==============================================================

from __future__ import annotations

import os
import pickle                                            # Loads the prepared sequences from prepare_data.py

import numpy as np                                        # Handles array/one-hot encoding operations
from tensorflow.keras.callbacks import ModelCheckpoint     # Saves the model whenever it improves during training
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Sequential              # Builds the neural network layer by layer
from tensorflow.keras.utils import to_categorical            # Converts integer labels into one-hot vectors

NOTES_FILE = "data/notes.pkl"                              # Where the prepared training data was saved
MODEL_DIR = "models"                                        # Folder where trained model checkpoints get saved
EPOCHS = 100                                                 # Number of full passes over the training data
BATCH_SIZE = 64                                              # Number of sequences processed before each weight update


def load_prepared_data(file_path: str) -> dict:
    """Load the prepared note sequences created by prepare_data.py."""
    with open(file_path, "rb") as notes_file:
        return pickle.load(notes_file)                       # Restore the dict of arrays/mappings saved earlier


def build_model(sequence_length: int, vocab_size: int) -> Sequential:
    """
    Build the LSTM network used to predict the next note in a sequence.

    Args:
        sequence_length: number of timesteps in each input sequence.
        vocab_size: number of unique notes/chords (size of the output layer).

    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential()                                                     # Start building the layers in order
    model.add(LSTM(512, input_shape=(sequence_length, 1), return_sequences=True))  # First LSTM layer, keeps sequence
    model.add(Dropout(0.3))                                                  # Randomly drop connections to reduce overfitting
    model.add(LSTM(512, return_sequences=True))                              # Second LSTM layer, keeps sequence
    model.add(Dropout(0.3))
    model.add(LSTM(512))                                                     # Final LSTM layer, outputs a single vector
    model.add(BatchNormalization())                                          # Stabilizes and speeds up training
    model.add(Dense(256))                                                    # Fully connected layer for extra learning capacity
    model.add(Dropout(0.3))
    model.add(Dense(vocab_size))                                             # Output layer: one node per possible note
    model.add(Activation("softmax"))                                         # Converts outputs into note probabilities

    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")      # Standard setup for multi-class prediction
    return model


def main() -> None:
    print("=" * 50)
    print("       MUSIC GENERATION - MODEL TRAINING")
    print("=" * 50)

    data = load_prepared_data(NOTES_FILE)                    # Load the sequences prepared in the previous stage
    network_input = data["network_input"]                    # Input sequences: shape (samples, timesteps, 1)
    network_output = data["network_output"]                  # Target notes as plain integers
    vocab_size = data["vocab_size"]                           # Total number of unique notes/chords
    sequence_length = data["sequence_length"]                 # Length of each input sequence

    network_output = to_categorical(network_output, num_classes=vocab_size)  # One-hot encode the target notes

    model = build_model(sequence_length, vocab_size)          # Build the LSTM network
    model.summary()                                           # Print the model architecture for reference

    os.makedirs(MODEL_DIR, exist_ok=True)                     # Create the models/ folder if it doesn't exist yet
    checkpoint_path = f"{MODEL_DIR}/weights-{{epoch:02d}}-{{loss:.4f}}.keras"
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="loss", verbose=1, save_best_only=True, mode="min"
    )                                                          # Save the model only when its loss improves

    print(f"\nTraining on {len(network_input)} sequences for {EPOCHS} epochs...")
    model.fit(
        network_input,
        network_output,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint],
    )                                                          # Train the model, saving checkpoints along the way

    final_model_path = f"{MODEL_DIR}/final_model.keras"
    model.save(final_model_path)                              # Save the fully trained model for the generation stage
    print(f"\nTraining complete. Final model saved to: {final_model_path}")
    print("Next step: run generate.py")


if __name__ == "__main__":
    main()   # Run the program only when executed directly (not when imported)
