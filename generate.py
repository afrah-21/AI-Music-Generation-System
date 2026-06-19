import pickle
import numpy as np
from music21 import note, chord, stream
from tensorflow.keras.models import load_model

model = load_model("music_model.h5")

with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

with open("pitchnames.pkl", "rb") as f:
    pitchnames = pickle.load(f)

n_vocab = len(pitchnames)
note_to_int = {note: number for number, note in enumerate(pitchnames)}
int_to_note = {number: note for number, note in enumerate(pitchnames)}

sequence_length = 100

network_input = []
for i in range(len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    network_input.append([note_to_int[n] for n in sequence_in])

start = np.random.randint(0, len(network_input) - 1)
pattern = network_input[start]

prediction_output = []

print("Generating music...")

for i in range(300):
    prediction_input = np.reshape(pattern, (1, sequence_length, 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)

    result = int_to_note[index]
    prediction_output.append(result)

    pattern.append(index)
    pattern = pattern[1:]

offset = 0
output_notes = []

for pattern in prediction_output:
    if "." in pattern or pattern.isdigit():
        notes_in_chord = pattern.split(".")
        chord_notes = []

        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            chord_notes.append(new_note)

        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset
        output_notes.append(new_chord)

    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write("midi", fp="generated_music.mid")

print("Music generated successfully!")
print("File saved as generated_music.mid")