import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from model import define_model

with open("artifacts/features.pkl", "rb") as f:
    features = pickle.load(f)

with open("artifacts/captions.pkl", "rb") as f:
    captions = pickle.load(f)

all_captions = []
for key in captions:
    all_captions.extend(captions[key])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in all_captions)

X1, X2, y = [], [], []

for img_id, caps in captions.items():
    for cap in caps:
        seq = tokenizer.texts_to_sequences([cap])[0]
        for i in range(1, len(seq)):
            in_seq = pad_sequences([seq[:i]], maxlen=max_length)[0]
            out_seq = to_categorical([seq[i]], num_classes=vocab_size)[0]
            X1.append(features[img_id][0])
            X2.append(in_seq)
            y.append(out_seq)

X1, X2, y = np.array(X1), np.array(X2), np.array(y)

model = define_model(vocab_size, max_length)
model.fit([X1, X2], y, epochs=10, batch_size=64)
model.save("saved_models/latest_model.h5")
