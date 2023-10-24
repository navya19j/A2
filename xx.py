from embeddings import *
from model import *
from utils import *
from data import *
import pickle
import torch
from torch import optim

def test_one_epoch(model, test_data_loader, label_mapping_inv, vocab_inv):
    for i, (sentence, sent_len, mask) in enumerate(test_data_loader):
        with torch.no_grad():

            sentence   = torch.LongTensor(sentence)

            tag_scores = model.predict(sentence, sent_len, mask)

            tag_scores = [item for sublist in tag_scores for item in sublist]

            # convert to tensor
            tag_scores = torch.LongTensor(tag_scores)

            # convert mask to 1D
            mask = mask.view(-1)

            outputs.append(tag_scores)

            # iterate over sentences
            for j in range(len(sentence)):

                # sentence 1
                sent1 = (sentence[j]).numpy()
                # remove pad
                
                pred1 = tag_scores[prev_idx:prev_idx+sent_len[j]].numpy()

                # tensor to int
                # remove 7535 from sent_

                # visualise
                sent_ = [vocab_inv[i] for i in sent1 if i!= 7535]
                pred_ = [label_mapping_inv[i] for i in pred1]

                # output to txt
                with open("output.txt", "w") as f:
                    for k in range(len(sent_)):
                        f.write(pred_[k] + "\n")
                    # add empty line
                    f.write("\n")

                prev_idx += sent_len[j]
    # accuracy
    outputs = torch.cat(outputs)

    return outputs

def test_pipeline(test_file, model_name , output_file):

    # get data
    print("Loading data...")
    test_sentences = get_test_data(test_file)

    model_checkpoint = model_name

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Loading model...")
    # load embeddings
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    print("Loading vocab...")
    # load vocab
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    print("Loading label mapping...")
    # load label mapping
    with open("label_mapping.pkl", "rb") as f:
        label_mapping = pickle.load(f)

    label_mapping_inv = {v:k for k,v in label_mapping.items()}

    # revert vocab 
    vocab_inv = {v:k for k,v in vocab.items()}

    model = LSTMTagCRF(embeddings, 256, len(vocab), len(label_mapping))

    print("Loading model checkpoint...")
    if(model_checkpoint is not None):
        # load model
        model.load_state_dict(torch.load(model_checkpoint))
        print("Loaded model from checkpoint")

    # convert to idx 
    test_sentences_idx = convert_to_idx_data(test_sentences, vocab)

    test_dataset  = NERdataset_Test(test_sentences_idx)

    test_data_loader = get_data_loader_test(test_dataset, vocab, BATCH_SIZE = 1)

    print("Testing...")

    for i, (sentence, sent_len, mask) in enumerate(test_data_loader):

        with torch.no_grad():

            sentence   = torch.LongTensor(sentence)

            tag_scores = model.predict(sentence, sent_len, mask)

            tag_scores = [item for sublist in tag_scores for item in sublist]

            # convert to tensor
            tag_scores = torch.LongTensor(tag_scores)

            # convert mask to 1D
            mask = mask.view(-1)

            # sentence 1
            
            tag_scores = tag_scores.numpy()
            sentence = sentence.numpy()
            sentence = sentence[0]

            # tensor to int
            # remove 7535 from sent_

            # visualise
            sent_ = [vocab_inv[i] for i in sentence if i!= len(vocab)]
            pred_ = [label_mapping_inv[i] for i in tag_scores]

            assert(len(sent_) == len(pred_))

            # output to txt
            with open(output_file, "a") as f:
                for k in range(len(sent_)):
                    f.write(pred_[k] + "\n")
                # add empty line
                f.write("\n")