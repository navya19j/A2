import torch

def save_checkpoint_v2(state, filename):
    torch.save(state, filename)


def load_checkpoint_v2(model, checkpoint):

    model.load_state_dict(checkpoint["state_dict"])

    return model

def load_optimizer_state(optimizer, checkpoint):

    optimizer.load_state_dict(checkpoint["optimizer"])

    return optimizer

def load_checkpoint_train(checkpoint, model, optimize):
    print("Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimize.load_state_dict(checkpoint["optimizer"])
    model.eval()