import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
# import pad sequnce
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF

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

def pad_collate(batch, pad_idx):

    (x, y) = zip(*batch)

    lens_x = [len(i) for i in x]

    x_pad = pad_sequence(x, batch_first=True, padding_value=pad_idx)
    y_pad = pad_sequence(y, batch_first=True, padding_value=-1)

    # where != pad_idx
    mask = (x_pad != pad_idx)

    return x_pad, y_pad, lens_x, mask

class LSTMTag(nn.Module):
    def __init__(self, pretrained_embedding, hidden_dim, vocab_size, target_size):
        super(LSTMTag, self).__init__()
        self.hidden_dim      = hidden_dim
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        self.lstm            = nn.LSTM(pretrained_embedding.shape[1], hidden_dim, bidirectional=True)
        self.dropout_layer   = nn.Dropout(p=0.5)
        self.hidden2tag      = nn.Linear(hidden_dim * 2, target_size)

    def forward(self, sentence, lengths):
        embeds      = self.word_embeddings(sentence)
        packed_seq  = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out    = self.dropout_layer(lstm_out)
        tag_space   = self.hidden2tag(lstm_out)
        tag_scores  =  F.log_softmax(tag_space, dim=1)

        return tag_scores

# add a CRF layer on top of it

class LSTMTagCRF(nn.Module):
    def __init__(self, pretrained_embedding, hidden_dim, vocab_size, target_size):
        super(LSTMTagCRF, self).__init__()
        self.hidden_dim      = hidden_dim
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        self.lstm            = nn.LSTM(pretrained_embedding.shape[1], hidden_dim, bidirectional=True)
        self.dropout_layer   = nn.Dropout(p=0.5)
        self.hidden2tag      = nn.Linear(hidden_dim * 2, target_size)
        self.crf             = CRF(target_size, batch_first = True)

    def compute_output(self, sentence, lengths):
        embeds      = self.word_embeddings(sentence)
        packed_seq  = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out    = self.dropout_layer(lstm_out)
        tag_space   = self.hidden2tag(lstm_out)
        
        return tag_space 

    def forward(self, sentence, lengths, labels,mask):

        emissions = self.compute_output(sentence,lengths)
        return -self.crf(emissions, labels, mask = mask)

    def predict(self, sentence,lengths, mask):

        scores = self.compute_output(sentence, lengths)

        return self.crf.decode(scores, mask=mask)