import torch

from .detection import DetectionTask
from .util import get_ckpt_callback, get_early_stop_callback
from .util import get_logger


# This method for getting the args without a checkpoint
def get_task(args):
    return DetectionTask(args)


# This method is implemented to take the args and a checkpoint
def get_trained_task(args, ckpt_path):
    return DetectionTask(args).load_from_checkpoint(checkpoint_path=ckpt_path)


# This method is for testing
def load_task(ckpt_path, **kwargs):
    args = torch.load(ckpt_path, map_location='cpu')['hyper_parameters']
    return DetectionTask.load_from_checkpoint(ckpt_path, **kwargs)
