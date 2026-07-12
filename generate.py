# ==============================================================
# CodeAlpha Artificial Intelligence Internship
# Project      : Music Generation with AI
# Developed By : Fahad Ahmad
# Student ID   : CA/DF1/189817
# Description  : Stage 3 of 3 - Music Generation.
#                Loads the trained LSTM model, feeds it a random seed
#                sequence from the training data, predicts a new
#                sequence of notes one step at a time, and converts the
#                result into a playable MIDI file using music21.
# ==============================================================

from __future__ import annotations

import pickle                                            # Loads the note vocabulary saved by prepare_data.py

import numpy as np                                        # Handles array reshaping and random seed selection
from music21 import instrument, note, chord, stream        # Converts predicted notes back into a MIDI file
from tensorflow.keras.models import load_model              # Loads the trained model saved by train.py

NOTES_FILE = "data/notes.pkl"                              # Prepared data file (contains the note vocabulary)
MODEL_PATH = "models/final_model.keras"                     # Path to the trained model
OUTPUT_MIDI = "generated_output.mid"                        # Filename for the generated music
NUM_NOTES_TO_GENERATE = 200                                 # How many new notes/chords to generate


def load_resources() -> tuple[dict, "keras.Model"]:
    """Load the saved note vocabulary and the trained model."""
    with open(NOTES_FILE, "rb") as notes_file:
        data = pickle.load(notes_file)                       # Restore vocabulary and sequence info

    model = load_model(MODEL_PATH)                           # Load the trained LSTM model from disk
    return data, model


def generate_notes(model, network_input: np.ndarray, note_to_int: dict, vocab_size: int) -> list[str]:
    """
    Generate a new sequence of notes using the trained model.

    Args:
        model: the trained Keras model.
        network_input: the original training input sequences (used to pick a random seed).
        note_to_int: mapping from note name to integer, used to build the reverse mapping.
        vocab_size: total number of unique notes/chords.

    Returns:
        A list of generated note/chord names.
    """
    int_to_note = {number: note_name for note_name, number in note_to_int.items()}  # Reverse the note->int mapping

    start_index = np.random.randint(0, len(network_input) - 1)     # Pick a random starting point from the dataset
    pattern = list(network_input[start_index] * vocab_size)         # Un-normalize back to integer note values
    pattern = [int(value[0]) for value in pattern]                   # Flatten to a plain list of integers

    generated_notes = []                                             # Will collect the newly generated notes

    for _ in range(NUM_NOTES_TO_GENERATE):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))  # Reshape into the model's expected input shape
        prediction_input = prediction_input / float(vocab_size)       # Normalize the same way training data was

        prediction = model.predict(prediction_input, verbose=0)       # Get the probability distribution over notes
        predicted_index = np.argmax(prediction)                       # Pick the most likely next note

        generated_notes.append(int_to_note[predicted_index])          # Convert the predicted integer back to a note
        pattern.append(predicted_index)                                # Add the prediction to the running sequence
        pattern = pattern[1:]                                          # Slide the window forward by one note

    return generated_notes


def create_midi(generated_notes: list[str], output_path: str) -> None:
    """
    Convert a list of note/chord names into a MIDI file.

    Args:
        generated_notes: list of note/chord name strings.
        output_path: where to save the resulting .mid file.
    """
    output_stream = stream.Stream()                                    # Container that will hold the music sequence
    offset = 0.0                                                       # Tracks how far into the piece each note sits

    for pattern in generated_notes:
        if "." in pattern:                                             # A "." means this pattern is a chord
            chord_notes = [note.Note(int(n)) for n in pattern.split(".")]
            new_chord = chord.Chord(chord_notes)                        # Rebuild the chord from its note numbers
            new_chord.offset = offset
            new_chord.storedInstrument = instrument.Piano()             # Assign a piano sound to the chord
            output_stream.append(new_chord)
        else:                                                          # Otherwise it's a single note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()              # Assign a piano sound to the note
            output_stream.append(new_note)

        offset += 0.5                                                  # Space each note/chord half a beat apart

    output_stream.write("midi", fp=output_path)                        # Write the finished sequence to a MIDI file


def main() -> None:
    print("=" * 50)
    print("       MUSIC GENERATION")
    print("=" * 50)

    data, model = load_resources()                                     # Load the vocabulary and trained model
    network_input = data["network_input"]
    note_to_int = data["note_to_int"]
    vocab_size = data["vocab_size"]

    print(f"Generating {NUM_NOTES_TO_GENERATE} new notes...")
    generated_notes = generate_notes(model, network_input, note_to_int, vocab_size)  # Generate the new sequence

    create_midi(generated_notes, OUTPUT_MIDI)                          # Save the generated sequence as a MIDI file
    print(f"\nDone! Generated music saved to: {OUTPUT_MIDI}")


if __name__ == "__main__":
    main()   # Run the program only when executed directly (not when imported)
