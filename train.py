from embeddings import *
from model import *
from utils import *
from data import *
import pickle
import torch
import torch.nn as nn
from torch import optim

def validate_one_epoch_crf( test_data_loader, model):
    outputs = []
    labels  = []

    for i, (sentence, label, sent_len, mask) in enumerate(test_data_loader):
        with torch.no_grad():

            sentence   = torch.LongTensor(sentence)

            label      = torch.LongTensor(label)

            # convert to 1d
            label      = label.view(label.shape[0]*label.shape[1])

            tag_scores = model.predict(sentence, sent_len, mask)

            tag_scores = [item for sublist in tag_scores for item in sublist]

            # convert to tensor
            tag_scores = torch.LongTensor(tag_scores)

            # convert mask to 1D
            mask = mask.view(-1)

            # remove pads
            # tag_scores = tag_scores[mask ==True]
            label      = label[mask == True]   

            outputs.append(tag_scores)
            labels.append(label)  

    # accuracy
    outputs = torch.cat(outputs)
    labels = torch.cat(labels)

    acc = torch.sum(outputs == labels).item() / len(labels)

    return acc

def train_one_epoch_baseline( train_data_loader, model, optimizer, loss_function):

    outputs = []
    labels = []
    losses = 0

    for i, (sentence, label, sent_len, mask) in enumerate(train_data_loader):

        optimizer.zero_grad()

        sentence = torch.LongTensor(sentence)

        label = torch.LongTensor(label)
        # convert to 1d
        label = label.view(label.shape[0]*label.shape[1])

        tag_scores = model(sentence, sent_len)

        # flatten
        tag_scores = tag_scores.view(-1, tag_scores.shape[-1])
        loss = loss_function(tag_scores, label)
        losses += loss

        # convert mask to 1D
        mask = mask.view(-1)

        # remove pads
        tag_scores = tag_scores[mask == True]
        label = label[mask == True]        

        pred_labels = torch.argmax(tag_scores, dim=1)
        
        loss.backward()
        optimizer.step()
        outputs.append(pred_labels)
        labels.append(label)

    # accuracy
    outputs = torch.cat(outputs)
    labels = torch.cat(labels)

    acc = torch.sum(outputs == labels).item() / len(labels)

    print("Accuracy: {}".format(acc))

    return acc,losses

def train_one_epoch_crf( train_data_loader, model, optimizer):

    outputs = []
    labels  = []
    losses  = 0

    for i, (sentence, label, sent_len, mask) in enumerate(train_data_loader):

        optimizer.zero_grad()

        sentence = torch.LongTensor(sentence)

        label    = torch.LongTensor(label)

        # get loss
        loss = model(sentence, sent_len, label,mask)

        losses += (loss.item()/(len(label[0])*len(sentence)))

        tag_scores = model.predict(sentence, sent_len,mask)

        tag_scores = [item for sublist in tag_scores for item in sublist]

        # convert to tensor
        tag_scores = torch.LongTensor(tag_scores)

        # flatten label
        label      = label.view(label.shape[0]*label.shape[1])
        # convert mask to 1D
        mask = mask.view(-1)

        # remove pads
        # tag_scores = tag_scores[mask ==True]
        label      = label[mask == True]     

        # pred_labels = torch.argmax(tag_scores, dim=1)
        
        loss.backward()
        optimizer.step()
        outputs.append(tag_scores)
        labels.append(label)

    # accuracy
    outputs = torch.cat(outputs)
    labels  = torch.cat(labels)
    acc     = torch.sum(outputs == labels).item() / len(labels)

    print("Accuracy: {}".format(acc))
    print("Loss: {}".format(losses))

    return acc,losses

def train_pipeline(train_file, labels_json, val_filename, model_name):
    
    # get data
    print("Loading data...")
    train_sentences, train_labels, label_mapping = get_data(file_name=train_file, label_file=labels_json)
    vocab                                        = get_vocab(train_sentences)

    # save vocabulary pickle
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    train_sentences_idx                          = convert_to_idx_data(train_sentences, vocab)
    train_labels_idx                             = convert_to_idx_labels(train_labels, label_mapping)

    print("Loading embeddings...")
    # get glove+pos embeddings
    embed_tensor = get_embed_matrix(vocab)

    # save embeddings
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embed_tensor, f)

    # save label mapping
    with open("label_mapping.pkl", "wb") as f:
        pickle.dump(label_mapping, f)

    batch_size    = 32

    # save vocab and label mapping to dict2
    test_sentences, test_labels, _ = get_data(file_name = val_filename)
    test_sentences_idx = convert_to_idx_data(test_sentences, vocab)
    test_labels_idx    = convert_to_idx_labels(test_labels, label_mapping)

    train_dataset = NERdataset(train_sentences_idx, train_labels_idx, train = True)
    test_dataset  = NERdataset(test_sentences_idx, test_labels_idx, train = False)

    train_data_loader , test_data_loader  = get_data_loader(train_dataset, test_dataset, vocab, BATCH_SIZE = batch_size)

    print("Train model")

    num_epochs    = 20
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    model         = LSTMTagCRF(embed_tensor, 256, len(vocab), len(label_mapping))
    optimizer     = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    num_batches   = len(train_sentences_idx) // batch_size

    prev_acc = 0
    i = 0

    for epoch in range(num_epochs):
        print("Epoch: {}".format(epoch))
        train_one_epoch_crf(train_data_loader, model, optimizer)
        acc = validate_one_epoch_crf(test_data_loader, model)
        print("Test accuracy: {}".format(acc))

        if acc > prev_acc:
            torch.save(model.state_dict(), model_name)
            prev_acc = acc
        else:
            i += 1
            if i == 5:
                break