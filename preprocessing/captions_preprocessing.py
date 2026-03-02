import string
import pickle

def load_captions(filename):
    captions = {}
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) < 2:
                continue
            img_id = tokens[0].split('#')[0]
            caption = tokens[1].lower()
            caption = caption.translate(str.maketrans('', '', string.punctuation))
            caption = 'startseq ' + caption + ' endseq'
            captions.setdefault(img_id, []).append(caption)
    return captions

def save_captions(captions, path):
    with open(path, 'wb') as f:
        pickle.dump(captions, f)
