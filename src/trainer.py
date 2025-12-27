from pathlib import Path
from typing import Dict
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchmetrics import Accuracy, F1Score, AUROC, ConfusionMatrix
#
from model import GenreClassifier
from music_genre.src.utils import get_run_name


class GenreClassificationModule(pl.LightningModule):
    """流派分类Lightning模块"""
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self.save_hyperparameters()
        # 模型
        self.model = GenreClassifier(config)
        # 损失函数
        self.loss_fun = nn.CrossEntropyLoss()
        # 训练指标
        num_classes = config.model.num_classes
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        # 验证集额外指标
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_auroc = AUROC(task='multiclass', num_classes=num_classes)
        # 测试集额外指标
        self.test_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.test_auroc = AUROC(task='multiclass', num_classes=num_classes)
        self.test_cm = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        # 用于记录最佳指标
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str):
        """
        """
        if isinstance(batch, list):
            batch = batch[0]
        if isinstance(batch, dict):
            # 这种判断方式更安全：检查第一个 value 是否还是字典
            first_val = next(iter(batch.values())) if batch else None
            if isinstance(first_val, dict):
                # 假设只有一个数据集，取第一个
                dataset_name = list(batch.keys())[0]
                batch = batch[dataset_name]
        if "x_mert" in batch:
            x = batch["x_mert"]
            labels = batch["y_genre"]
        else:
            x = batch["x"]
            labels = batch["label"]
        # 前向传播
        logits = self(x)
        loss = self.loss_fun(logits, labels)
        # 计算概率（用于AUROC）
        probs = torch.softmax(logits, dim=1)
        # 记录指标
        batch_size = x.size(0)
        if stage == 'train':
            self.train_acc(logits, labels)
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        elif stage == 'val':
            self.val_acc(logits, labels)
            self.val_f1(logits, labels)
            self.val_auroc(probs, labels)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, batch_size=batch_size)
        elif stage == 'test':
            self.test_acc(logits, labels)
            self.test_f1(logits, labels)
            self.test_auroc(probs, labels)
            self.test_cm(logits, labels)
            self.log('test_loss', loss, batch_size=batch_size)
            self.log('test_acc', self.test_acc, batch_size=batch_size)
            self.log('test_f1', self.test_f1, batch_size=batch_size)
            self.log('test_auroc', self.test_auroc, batch_size=batch_size)
        return loss,logits,labels

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss,_,_=self._shared_step(batch,'train')
        return loss
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._shared_step(batch, 'val')
        return loss
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss, logits, labels = self._shared_step(batch, 'test')
        probs = torch.softmax(logits, dim=1)
        self.test_step_outputs.append({
            'probs': probs.cpu(),
            'labels': labels.cpu()
        })
        return loss
    def on_validation_epoch_end(self):
        current_acc = self.val_acc.compute()
        current_f1 = self.val_f1.compute()
        if current_acc > self.best_val_acc:
            self.best_val_acc = current_acc.item()
            self.log('best_val_acc', self.best_val_acc, prog_bar=False)
        if current_f1 > self.best_val_f1:
            self.best_val_f1 = current_f1.item()
            self.log('best_val_f1', self.best_val_f1, prog_bar=False)

    def on_test_epoch_end(self):
        acc = self.test_acc.compute().item()
        f1 = self.test_f1.compute().item()
        auroc = self.test_auroc.compute().item()


        self.test_results = {
            "acc": acc,
            "f1": f1,
            "auroc": auroc
        }

        dataset_name = self.config.data.active_datasets[0]
        run_name = get_run_name(
            self.config.feature_extraction.model_name,
            self.config.feature_extraction.segment_duration
        )
        save_dir = Path(self.config.output.result_dir) / dataset_name/run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.test_step_outputs:
            all_probs = torch.cat([x['probs'] for x in self.test_step_outputs])
            all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])

            save_path = save_dir / 'test_predictions.pt'

            torch.save({
                'probs': all_probs,  # 预测概率 [N, num_classes]
                'labels': all_labels,  # 真实标签 [N]
                'metrics': self.test_results,  # 整体指标
                'classes': self.config.data[dataset_name].classes if hasattr(self.config.data, dataset_name) else None
            }, save_path)

            print(f"\n✅ Test predictions and metrics saved to: {save_path}")
        else:
            print("\n⚠️ No test outputs collected.")

        self.test_step_outputs.clear()
    def configure_optimizers(self):
        optimizer_config = self.config.training.optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay
        )
        scheduler_config = self.config.training.scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.mode,
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": scheduler_config.monitor,
                "interval": "epoch",
                "frequency": 1
            }
        }