import numpy as np
import json
import torch
# import pad sequnce
from torch.nn.utils.rnn import pad_sequence

class NERdataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, train = True):
        self.sentences = sentences
        self.labels = labels

        if train:
            # sort by length
            self.sentences, self.labels = (list(t) for t in zip(*sorted(zip(self.sentences, self.labels), key=lambda x: len(x[0]), reverse=True)))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        # return a tensor
        return torch.tensor(self.sentences[index]), torch.tensor(self.labels[index])

class NERdataset_Test(torch.utils.data.Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        # return a tensor
        return torch.tensor(self.sentences[index])

def pad_collate(batch, pad_idx):

    (x, y) = zip(*batch)

    lens_x = [len(i) for i in x]

    x_pad = pad_sequence(x, batch_first=True, padding_value=pad_idx)
    y_pad = pad_sequence(y, batch_first=True, padding_value=-1)

    # where != pad_idx
    mask = (x_pad != pad_idx)

    return x_pad, y_pad, lens_x, mask

def pad_collate_test(batch, pad_idx):

    lens_x = [len(i) for i in batch]

    x_pad = pad_sequence(batch, batch_first=True, padding_value=pad_idx)

    # where != pad_idx
    mask = (x_pad != pad_idx)

    return x_pad, lens_x, mask

def get_data_loader(train_dataset,test_dataset,vocab, BATCH_SIZE = 32):
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: pad_collate(x, vocab['pad_idx']))
    test_data_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: pad_collate(x, vocab['pad_idx']))

    return train_data_loader, test_data_loader

def get_data_loader_test(test_dataset,vocab, BATCH_SIZE = 1):
    test_data_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: pad_collate_test(x, vocab['pad_idx']))

    return test_data_loader

def get_data(file_name = "train.txt" , label_file = "labels.json" ):
    train_sentences = []
    train_labels = []
    sentence = ""
    label = ""

    with open(file_name, "r" ) as f:
        lines = f.readlines()
    for line in lines:
        if (line == "\n"):
            train_sentences.append(sentence[0:len(sentence)-1])
            train_labels.append(label[0:len(label)-1])
            sentence = ""
            label = ""
        else:
            sentence += line.split()[0] + " "
            label += line.split()[1] + " "

    # get labels from labels.json
    with open(label_file, "r") as f:
        labels = json.load(f)

    labels["pad_idx"] = -1

    return train_sentences, train_labels, labels

def get_test_data(file_name = "test.txt" ):
    test_sentences = []
    sentence = ""

    with open(file_name, "r" ) as f:
        lines = f.readlines()

    for line in lines:
        if (line == "\n"):
            test_sentences.append(sentence[0:len(sentence)-1])
            sentence = ""
        else:
            sentence += line.split()[0] + " "

    print(len(test_sentences))

    return test_sentences

def get_glove(path = "embeddings/glove.6B.200d.txt"):
    # get GloVe embeddings
    embeddings_index = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            word,vector = line.split(maxsplit=1)
            vector = vector.strip("\n")
            # convert string to np array
            vector = np.fromstring(vector, dtype=np.float32, sep=" ")
            embeddings_index[word] = vector

    # initialise unk randomly
    embeddings_index["unk"] = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), 200)
    # embeddings_index["unk"] = np.zeros(100)
    embeddings_index["pad_idx"] = np.zeros(200)

    # add chemical embeddings

    return embeddings_index