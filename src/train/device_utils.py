import torch


def get_device() -> torch.device:
    device: torch.device = None

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("[Device] " + str(device))

    return device


if __name__ == "__main__":
    d = get_device()
