import sys
from train import *
from test import *

# main
if __name__ == "__main__":

    mode = sys.argv[1]

    if mode == "train":

        TRAIN_FILE = sys.argv[2]
        DEV_FILE = sys.argv[3]

        LABELS_FILE = "labels.json"
        MODEL_FILE  = "model.pt"

        # run training
        train_pipeline(train_file=TRAIN_FILE, labels_json=LABELS_FILE, val_filename=DEV_FILE, model_name=MODEL_FILE)

    elif mode == "test":

        TEST_FILE  = sys.argv[2]
        MODEL_FILE = "model.pt"

        if (len(sys.argv) > 3):
            OUTPUT_FILE = sys.argv[3]
        else:
            OUTPUT_FILE = "outputfile.txt"

        # run testing
        test_pipeline(test_file=TEST_FILE, model_name=MODEL_FILE, output_file=OUTPUT_FILE)