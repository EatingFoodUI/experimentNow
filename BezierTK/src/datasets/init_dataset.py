from .sample.sampler import Sampler

from .dataset.MOT17 import MOT17
from .dataset.MOT20 import MOT20


switch_dataset = {
    'mot17': MOT17,
    'mot20': MOT20,
}

# ？？
def get_dataset(dataset):
    class Dataset(switch_dataset[dataset], Sampler):
        pass

    return Dataset
