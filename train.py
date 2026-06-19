import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

print("Loading notes...")

with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

sequence_length = 100

pitchnames = sorted(set(notes))
n_vocab = len(pitchnames)

note_to_int = {note: number for number, note in enumerate(pitchnames)}

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]

    network_input.append([note_to_int[n] for n in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

print("Patterns:", n_patterns)

network_input = np.reshape(
    network_input,
    (n_patterns, sequence_length, 1)
)

network_input = network_input / float(n_vocab)

network_output = to_categorical(network_output)

model = Sequential()

model.add(
    LSTM(
        128,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(LSTM(128))

model.add(Dense(128, activation="relu"))

model.add(Dense(n_vocab, activation="softmax"))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)

model.summary()

model.fit(
    network_input,
    network_output,
    epochs=10,
    batch_size=32
)

model.save("music_model.h5")

with open("pitchnames.pkl", "wb") as f:
    pickle.dump(pitchnames, f)

print("Training Complete!")