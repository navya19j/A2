from features import *
from data import *
import numpy as np
import re
import torch

# fix random seed
np.random.seed(42)

# get vector for each word in the vocabulary
# if word is not in GloVe, use unk
def get_vector(word, embeddings_index):

    # normal embedding
    if has_multiple_words(word):
        if("/" in word):
            # split by "/"
            words = word.split("/")
            word = words[0]
        elif("-" in word):
            # split by "-"
            words = word.split("-")
            word = words[0]

    if word in embeddings_index:
        return embeddings_index[word]

    elif word.lower() in embeddings_index:
        return embeddings_index[word.lower()]

    elif lemma(word) in embeddings_index:
        return embeddings_index[lemma(word)]
        
    elif is_quantity(word):
        # split quantity into number and unit
        number = re.findall(r'\d+', word)[0]
        unit   = re.findall(r'[a-zA-Z]+', word)[0]
        # get embedding for number and unit
        if number in embeddings_index:
            number_embedding = embeddings_index[number]
        else:
            number_embedding = embeddings_index["number"]
        
        unit_embedding = embeddings_index[unit] if unit in embeddings_index else embeddings_index["quantity"]
        
        # return sum of number and unit embeddings
        return (number_embedding + unit_embedding)

    elif(is_chemical(word)):
        return embeddings_index["chemical"]

    elif(is_device(word)):
        return embeddings_index["device"]

    elif (isAllUpper(word)):
        return embeddings_index["acronym"]
    
    elif(preprocess(word) in embeddings_index):
        return embeddings_index[preprocess(word)]

    else:
        return embeddings_index["unk"]

def pos_tag_vocab(vocab):
    # create pos tag vecs
    pos_tag_idx = {}
    i = 0
    for word in vocab:
        # get pos tag
        try:
            pos_tag = get_pos_tag(word)
            # get vector for pos tag
            if (pos_tag in pos_tag_idx):
                continue
            else:
                pos_tag_idx[pos_tag] = i
                i += 1

        except:
            print("Error: ", word)

    return pos_tag_idx

def get_combined_vector(word,pos_tag_idx,embeddings_index):
    
    glove_vec = get_vector(word,embeddings_index)
    pos_tag = get_pos_tag(word)
    pos_tag_vec = pos_tag_idx[pos_tag]
    is_capital_vec = 1 if isFirstUpper(word) else 0
    is_all_capital_vec = 1 if isAllUpper(word) else 0
    is_mixed_capital_vec = 1 if isMixedcaps(word) else 0
    is_number_vec = 1 if number(word) else 0
    is_quantity_vec = 1 if is_quantity(word) else 0
    is_chemical_vec = 1 if is_chemical(word) else 0
    is_punctuation_vec = 1 if has_punctuation(word) else 0
    isdevice = 1 if is_device(word) else 0

    return np.concatenate((glove_vec, [ pos_tag_vec , is_capital_vec, is_all_capital_vec, is_mixed_capital_vec, is_number_vec, is_quantity_vec, is_chemical_vec, is_punctuation_vec,isdevice]), axis=0)

def get_vocab(sentences):
    # get vocabulary
    vocab = {}
    for sentence in sentences:
        for word in sentence.split(" "):
            word = preprocess(word)
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1

    # sort by frequency
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    vocab_dict = {}
    # revert dictionary starting with 1
    i = 0
    for word in vocab:
        vocab_dict[word[0]] = i
        i += 1

    vocab_dict["unk"] = i
    i += 1
    vocab_dict["pad_idx"] = i

    return vocab_dict

def get_embed_matrix(vocab):
    # create embedding matrix
    embedding_matrix = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(vocab)+1, 209 ))
    embeddings_index = get_glove()
    pos_tag_idx = pos_tag_vocab(vocab)

    # change index of word embedding to match index of vocab    
    for word in vocab:
        embedding_matrix[vocab[word]] = get_combined_vector(word,pos_tag_idx, embeddings_index)

    weight_tensor = torch.FloatTensor(embedding_matrix)

    return weight_tensor

def get_embedding_labels(labels, label):
    embed = ""
    for l in label.split():
        embed += str(labels[l]) + " "
    return embed[0:len(embed)-1]

# general procedure
# pad = all 0'

def pre_process(key):

    key = key.lower()
    # lemmatize
    key = lemma(key)

    return key

def pre_processed(vocab):

    old_keys = list(vocab.keys())
    new_keys = []

    for key in old_keys:
        new_keys.append(pre_process(key))

    # make dict
    new_vocab = {}
    for i in range(len(old_keys)):
        new_vocab[new_keys[i]] = vocab[old_keys[i]]

    return new_vocab

def convert_to_idx_data(train_sentences,vocab):
    # convert train sentences to indexes
    train_sentences_idx = []
    pre_processed_vocab = pre_processed(vocab)

    for sentence in train_sentences:
        sentence_idx = []
        for word in sentence.split():
            word = preprocess(word)
            if word in vocab:
                sentence_idx.append(vocab[word])
            else:
                # check if word is in pre_processed vocab
                if word in pre_processed_vocab:
                    sentence_idx.append(pre_processed_vocab[word])
                else:
                    sentence_idx.append(vocab["unk"])
        train_sentences_idx.append(sentence_idx)

    return train_sentences_idx

def convert_to_idx_labels(train_labels,labels):
    # convert train labels to indexes
    # convert labels to index
    train_labels_idx = []
    for label in train_labels:
        str_labels = get_embedding_labels(labels,label)
        # spli by space convert to list
        train_labels_idx.append(list(map(int, str_labels.split())))

    return train_labels_idx

def get_vocab_chars(words):
    # get vocabulary
    vocab = {}
    for word in words:
        for char in word:
            char = preprocess(char)
            if char in vocab:
                vocab[char] += 1
            else:
                vocab[char] = 1
    
    # sort by frequency
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    vocab_dict = {}
    # revert dictionary starting with 1
    i = 0
    for word in vocab:
        vocab_dict[word[0]] = i
        i += 1

    return vocab_dict

def add_padding(sentence, labels , max_len, vocab):
    while (len(sentence) < max_len):
        sentence = sentence.append(vocab["pad_idx"])
        labels = labels.append(labels["pad_idx"])

    return sentence, labels

def char_embed(train_sentences,vocab):
    # convert each sentence to list of lists of chars
    unique_chars = get_vocab_chars(vocab)
    train_sentences_chars = []
    for sentence in train_sentences:
        sentence_chars = []
        for word in sentence.split():
            word_chars = []
            for char in word:
                char = preprocess(char)
                if char in unique_chars:
                    word_chars.append(unique_chars[char])
                else:
                    word_chars.append(unique_chars["unk"])
            sentence_chars.append(word_chars)
        train_sentences_chars.append(sentence_chars)