# ==============================================================
# CodeAlpha Artificial Intelligence Internship
# Project      : Music Generation with AI
# Developed By : Fahad Ahmad
# Student ID   : CA/DF1/189817
# Description  : Stage 1 of 3 - Data Preparation.
#                Scans a folder of MIDI files, extracts every note and
#                chord using music21, builds a vocabulary mapping notes
#                to integers, and saves fixed-length input/output
#                sequences ready for training an LSTM model.
# ==============================================================

from __future__ import annotations

import glob                                    # Finds all MIDI file paths inside the dataset folder
import pickle                                  # Saves prepared Python objects to disk for later use

import numpy as np                             # Handles the numeric arrays used for model training
from music21 import converter, instrument, note, chord   # Parses MIDI files and reads notes/chords

MIDI_FOLDER = "midi_songs"                     # Folder containing the training .mid files
NOTES_FILE = "data/notes.pkl"                  # Where the extracted raw notes/chords get saved
SEQUENCE_LENGTH = 100                          # How many previous notes the model looks at to predict the next one


def extract_notes_from_midi(midi_folder: str) -> list[str]:
    """
    Parse every MIDI file in a folder and extract its notes and chords.

    Args:
        midi_folder: path to the folder containing .mid files.

    Returns:
        A flat list of note/chord names across all songs, in order.
    """
    notes = []                                              # Will hold every note/chord found, across all songs
    midi_files = glob.glob(f"{midi_folder}/*.mid")           # Find every MIDI file in the dataset folder

    if not midi_files:
        raise FileNotFoundError(
            f"No .mid files found in '{midi_folder}'. Add some MIDI files before running this script."
        )

    for file_path in midi_files:
        print(f"Parsing: {file_path}")                       # Show progress while processing many files
        midi_stream = converter.parse(file_path)             # Load the MIDI file into a music21 stream

        try:
            parts = instrument.partitionByInstrument(midi_stream)  # Split multi-instrument tracks apart
            elements = parts.parts[0].recurse() if parts else midi_stream.flat.notes
        except Exception:
            elements = midi_stream.flat.notes                # Fall back to a flat note list if parsing fails

        for element in elements:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))             # Store single notes as their pitch, e.g. "C4"
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))  # Store chords as joined note numbers

    return notes


def build_sequences(notes: list[str]) -> tuple[np.ndarray, np.ndarray, dict, int]:
    """
    Convert the raw note list into numeric input/output training sequences.

    Args:
        notes: flat list of note/chord names.

    Returns:
        A tuple of (network_input, network_output, note_to_int mapping, vocabulary size).
    """
    pitch_names = sorted(set(notes))                          # Unique notes/chords across the whole dataset
    note_to_int = {note_name: number for number, note_name in enumerate(pitch_names)}  # Map each note to an integer
    vocab_size = len(pitch_names)                              # Total number of unique notes/chords

    network_input = []                                         # Will hold sequences of SEQUENCE_LENGTH notes
    network_output = []                                        # Will hold the single "next note" for each sequence

    for i in range(len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i : i + SEQUENCE_LENGTH]           # A window of SEQUENCE_LENGTH consecutive notes
        sequence_out = notes[i + SEQUENCE_LENGTH]              # The note that comes right after that window
        network_input.append([note_to_int[n] for n in sequence_in])   # Convert the window to integers
        network_output.append(note_to_int[sequence_out])       # Convert the target note to an integer

    n_patterns = len(network_input)                            # How many training examples we ended up with

    # Reshape and normalize input for the LSTM: (samples, timesteps, features)
    network_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    network_input = network_input / float(vocab_size)          # Scale values to the 0-1 range for stable training
    network_output = np.array(network_output)                  # Keep the output as plain integer class labels

    return network_input, network_output, note_to_int, vocab_size


def main() -> None:
    print("=" * 50)
    print("       MUSIC DATA PREPARATION")
    print("=" * 50)

    notes = extract_notes_from_midi(MIDI_FOLDER)               # Extract every note/chord from all MIDI files
    print(f"\nExtracted {len(notes)} notes/chords from the dataset.")

    network_input, network_output, note_to_int, vocab_size = build_sequences(notes)
    print(f"Built {len(network_input)} training sequences.")
    print(f"Vocabulary size: {vocab_size} unique notes/chords.")

    import os
    os.makedirs("data", exist_ok=True)                          # Create the data/ folder if it doesn't exist yet

    with open(NOTES_FILE, "wb") as notes_file:
        pickle.dump(
            {
                "notes": notes,
                "network_input": network_input,
                "network_output": network_output,
                "note_to_int": note_to_int,
                "vocab_size": vocab_size,
                "sequence_length": SEQUENCE_LENGTH,
            },
            notes_file,
        )                                                       # Save everything needed for the training stage

    print(f"\nSaved prepared data to: {NOTES_FILE}")
    print("Next step: run train.py")


if __name__ == "__main__":
    main()   # Run the program only when executed directly (not when imported)
