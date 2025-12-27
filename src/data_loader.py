import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig
#
from dataset import DATASET_REGISTRY

class MusicDataModule(pl.LightningDataModule):
    """音乐数据模块 - 支持多数据集"""
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

    def setup(self, stage: str = None):
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

        active_datasets = self.config.data.active_datasets
        for dataset_name in active_datasets:
            if dataset_name not in DATASET_REGISTRY:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found in registry. "
                    f"Available: {list(DATASET_REGISTRY.keys())}"
                )

            dataset_class = DATASET_REGISTRY[dataset_name]
            # 创建训练/验证/测试数据集
            self.train_datasets.append(
                dataset_class(self.config, split='train')
            )
            self.val_datasets.append(
                dataset_class(self.config, split='val')
            )
            self.test_datasets.append(
                dataset_class(self.config, split='test')
            )

    def _create_dataloader(self, datasets, shuffle: bool) -> list:
        """创建数据加载器列表"""
        loaders = []

        for dataset in datasets:
            dataset_name = dataset.dataset_name
            dataset_config = getattr(self.config.data, dataset_name)

            loader = DataLoader(
                dataset,
                batch_size=dataset_config.batch_size,
                shuffle=shuffle,
                num_workers=dataset_config.num_workers,
                pin_memory=dataset_config.pin_memory,
                persistent_workers=dataset_config.num_workers > 0
            )
            loaders.append(loader)
        return loaders

    def train_dataloader(self):
        return self._create_dataloader(self.train_datasets, shuffle=True)
    def val_dataloader(self):
        return self._create_dataloader(self.val_datasets, shuffle=False)
    def test_dataloader(self):
        return self._create_dataloader(self.test_datasets, shuffle=False)