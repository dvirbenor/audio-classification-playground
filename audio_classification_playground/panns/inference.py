import numpy as np
import torch
from torch import nn

from .models import Cnn14_DecisionLevelMax
from .pytorch_utils import move_data_to_device
from .config import num_classes,labels

DEFAULT_SAMPLE_RATE = 32000
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_HOP_SIZE = 320
DEFAULT_MEL_BINS = 64
DEFAULT_FMIN = 50
DEFAULT_FMAX = 14000

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_model(
        checkpoint_path: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        window_size: int = DEFAULT_WINDOW_SIZE,
        hop_size: int = DEFAULT_HOP_SIZE,
        mel_bins: int = DEFAULT_MEL_BINS,
        fmin: int = DEFAULT_FMIN,
        fmax: int = DEFAULT_FMAX,
        classes_num: int = num_classes,
) -> nn.Module:
    model = Cnn14_DecisionLevelMax(sample_rate=sample_rate, window_size=window_size,
                                   hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                                   classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')

    return model


def frame_level_inference(model: nn.Module, audio: np.ndarray) -> np.ndarray:
    waveform = audio[None, :]

    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    return framewise_output




