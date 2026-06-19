import os
import pickle
from music21 import converter, instrument, note, chord

DATASET_PATH = "maestro-v3.0.0"
OUTPUT_FILE = "notes.pkl"

MAX_FILES = 30   # Only process 30 MIDI files for faster training

notes = []
count = 0

print("Reading MIDI files...")

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:

        if count >= MAX_FILES:
            break

        if file.endswith(".mid") or file.endswith(".midi"):
            file_path = os.path.join(root, file)
            print("Processing:", file_path)

            try:
                midi = converter.parse(file_path)
                parts = instrument.partitionByInstrument(midi)

                if parts:
                    notes_to_parse = parts.parts[0].recurse()
                else:
                    notes_to_parse = midi.flatten().notes

                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))

                    elif isinstance(element, chord.Chord):
                        notes.append(".".join(str(n) for n in element.normalOrder))

                count += 1
                print("Files processed:", count)

            except Exception as e:
                print("Error reading file:", file_path)
                print(e)

    if count >= MAX_FILES:
        break

print("Total files processed:", count)
print("Total notes extracted:", len(notes))

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(notes, f)

print("Notes saved successfully in notes.pkl")