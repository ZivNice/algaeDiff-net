import random
from torch.utils.data import Sampler
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class SeedRatioSampler(Sampler):
    def __init__(self, dataset, ratio=0.2, seed=42):
        self.dataset = dataset
        self.ratio = ratio
        self.seed = seed
        random.seed(self.seed)
        self.num_samples = int(len(dataset) * ratio)

    def __iter__(self):
        indices = random.sample(range(len(self.dataset)), k=self.num_samples)
        return iter(indices)

    def __len__(self):
        return self.num_samples
