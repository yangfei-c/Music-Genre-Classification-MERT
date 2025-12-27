"""
加载提取好多MERT特征
读取数据集的json文件进行划分，并对标签进行映射
"""
import json
import os
import random
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
#
from music_genre.src.utils import get_feature_dir


class BaseMusicDataset(Dataset):
    def __init__(self,config: DictConfig,split: str = 'train',dataset_name: str = None):
        self.config = config
        self.split = split
        self.dataset_name = dataset_name
        self.feature_dir = None
        self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def load_feature(self, feature_path: str) -> torch.Tensor:
        '''
        load_feature-->if_exist-->flatten
        '''
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        feature = np.load(feature_path)
        feature = torch.from_numpy(feature).float().squeeze(0)
        selected_layers = self.config.model.selected_layers
        selected_feature = feature[selected_layers, :]  # [num_selected, time, dim]
        flattened = selected_feature.flatten()
        return flattened

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement __getitem__")


class GtzanDataset(BaseMusicDataset):
    def __init__(self, config: DictConfig, split: str = 'train'):
        super().__init__(config, split, dataset_name='gtzan')
        self.dataset_config = config.data.gtzan
        self.feature_dir = get_feature_dir(
            base_dir=self.dataset_config.feature_dir,
            model_name_full=config.feature_extraction.model_name,
            duration=config.feature_extraction.segment_duration
        )
        self.meta_file = Path(self.dataset_config.meta_file)
        # 类别映射
        self.classes = self.dataset_config.classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        # 加载并划分数据
        self._load_and_split_data()
        #查看数据集划分逻辑
        #print(f"✅ [{split.upper()}] GTZAN Dataset: {len(self.samples)} samples")

    def _load_and_split_data(self):
        '''
        data-->split
        '''
        if not self.meta_file.exists():
            raise FileNotFoundError(f"Meta file not found: {self.meta_file}")
        with open(self.meta_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        random.seed(self.config.project.seed)
        indices = list(range(len(all_data)))
        random.shuffle(indices)
        total = len(indices)
        split_ratio = self.dataset_config.split_ratio
        train_end = int(total * split_ratio.train)
        val_end = train_end + int(total * split_ratio.val)

        if self.split == 'train':
            selected_indices = indices[:train_end]
        elif self.split == 'val':
            selected_indices = indices[train_end:val_end]
        elif self.split == 'test':
            selected_indices = indices[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        # 构建样本列表
        self.samples = [all_data[i] for i in selected_indices]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample = self.samples[idx]
        # 获取文件信息
        file_id = sample['id']
        genre = sample['genre']
        if genre not in self.class_to_idx:
            raise ValueError(f"Unknown genre: {genre}")
        label = torch.tensor(self.class_to_idx[genre], dtype=torch.long)

        file_name = Path(file_id).stem
        feature_path = self.feature_dir / file_name / "segment_0.npy"
        # 如果路径不存在，尝试备选路径
        if not feature_path.exists():
            feature_path = self.feature_dir / file_id / "segment_0.npy"

        try:
            x = self.load_feature(str(feature_path))
        except FileNotFoundError as e:
            print(f"⚠️ Warning: {e}")
            # 返回零特征作为fallback
            expected_dim = len(self.config.model.selected_layers) * self.config.model.layer_dim
            x = torch.zeros(expected_dim, dtype=torch.float32)

        return {
            "x": x,
            "label": label,
            "file_id": file_id
        }


class FMASmallDataset(BaseMusicDataset):
    def __init__(self, config: DictConfig, split: str = 'train'):
        # 初始化，设置 dataset_name 为 fma_small
        super().__init__(config, split, dataset_name='fma_small')
        self.dataset_config = config.data.fma_small
        self.feature_dir = get_feature_dir(
            base_dir=self.dataset_config.feature_dir,
            model_name_full=config.feature_extraction.model_name,
            duration=config.feature_extraction.segment_duration
        )
        self.meta_file = Path(self.dataset_config.meta_file)

        self.classes = self.dataset_config.classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self._load_and_split_data()

    def _load_and_split_data(self):
        if not self.meta_file.exists():
            raise FileNotFoundError(f"Meta file not found: {self.meta_file}")
        with open(self.meta_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        split_map = {
            'train': 'training',
            'val': 'validation',
            'test': 'test'
        }

        if self.split not in split_map:
            raise ValueError(f"Invalid split: {self.split}")

        target_split = split_map[self.split]

        self.samples = [
            item for item in all_data
            if item.get('split') == target_split
        ]
        if not self.samples:
            print(f"⚠️ Warning: No samples found for split '{self.split}' (looking for '{target_split}')")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        genre = sample['genre']
        if genre not in self.class_to_idx:
            raise ValueError(f"Unknown genre: {genre}. Please add it to config classes.")
        label = torch.tensor(self.class_to_idx[genre], dtype=torch.long)

        file_id = sample.get('id')
        file_path_str = sample.get('path', str(file_id))  # 如果没有 path 字段，回退到 ID
        file_stem = Path(file_path_str).stem  # 获取 "000002"
        feature_path = self.feature_dir / file_stem / "segment_0.npy"

        try:
            x = self.load_feature(str(feature_path))
        except FileNotFoundError:
            print(f"⚠️ Warning: Feature not found for {file_stem}")
            # 返回零向量作为 fallback，防止训练中断
            expected_dim = len(self.config.model.selected_layers) * self.config.model.layer_dim
            x = torch.zeros(expected_dim, dtype=torch.float32)

        return {
            "x": x,
            "label": label,
            "file_id": str(file_id)
        }
# 数据集注册表 - 方便扩展新数据集
DATASET_REGISTRY = {
    "gtzan": GtzanDataset,
     "fma_small": FMASmallDataset,
    # "million_song": MillionSongDataset,
}