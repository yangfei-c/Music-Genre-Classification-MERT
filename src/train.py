"""
训练脚本
使用PyTorch Lightning训练音乐流派分类模型
"""
import os
import warnings
from pathlib import Path
import logging
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger

from music_genre.src.utils import get_run_name

# 基础设置
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
#
from data_loader import MusicDataModule
from trainer import GenreClassificationModule

def setup_callbacks(config: DictConfig) -> list:
    callbacks = []
    callbacks.append(TQDMProgressBar(refresh_rate=20))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    # 早停
    if 'early_stopping' in config.training:
        es_config = config.training.early_stopping
        callbacks.append(
            EarlyStopping(
                monitor=es_config.monitor,
                patience=es_config.patience,
                mode=es_config.mode,
                verbose=False
            )
        )
    return callbacks

def print_config_summary(config: DictConfig):
    """打印配置摘要"""
    print("=" * 70)
    print("🎵 Music Genre Classification Training")
    print("=" * 70)
    print(">>Dataset Configuration:")
    print(f"   Active datasets: {', '.join(config.data.active_datasets)}")
    for dataset in config.data.active_datasets:
        ds_config = getattr(config.data, dataset)
        print(f"   {dataset.upper()}:")
        print(f"      Batch size: {ds_config.batch_size}")
        print(f"      Num workers: {ds_config.num_workers}")
        print(f"      Classes: {len(ds_config.classes)}"+'\n')

    print(">>Model Configuration:")
    print(f"   Selected MERT layers: {config.model.selected_layers}")
    print(f"   Layer dimension: {config.model.layer_dim}")
    print(f"   Hidden dimensions: {config.model.classifier.hidden_dims}")
    print(f"   Dropout: {config.model.classifier.dropout}")
    print(f"   Layer weighting: {config.model.classifier.use_layer_weighting}")
    print(f"   Num classes: {config.model.num_classes}"+'\n')

    print(">>Training Configuration:")
    print(f"   Max epochs: {config.training.max_epochs}")
    print(f"   Learning rate: {config.training.optimizer.lr}")
    print(f"   Weight decay: {config.training.optimizer.weight_decay}")
    print(f"   Checkpoint monitor: {config.training.checkpoint.monitor}")
    print(f"   Early stopping: {config.training.early_stopping.monitor} "
          f"(patience={config.training.early_stopping.patience})"+'\n')

    print(">>Output Paths:")
    print(f"   Checkpoints: {config.training.checkpoint.save_dir}")
    print(f"   TensorBoard: {config.output.tensorboard_dir}")
    print("=" * 70+'\n')

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    if config.data.active_datasets:
        current_dataset_name = config.data.active_datasets[0]
        if hasattr(config.data, current_dataset_name):
            dataset_cfg = getattr(config.data, current_dataset_name)

            # 自动计算类别列表的长度
            auto_num_classes = len(dataset_cfg.classes)

            print(f"🔄 Auto-detected dataset: [{current_dataset_name}]")
            print(f"🔄 Overriding num_classes: {config.model.get('num_classes', 'None')} -> {auto_num_classes}")

            with open_dict(config):
                config.model.num_classes = auto_num_classes
        else:
            raise ValueError(f"Config for dataset '{current_dataset_name}' not found in data section!")
    print_config_summary(config)
    pl.seed_everything(config.project.seed, workers=True)

    Path(config.output.tensorboard_dir).mkdir(parents=True, exist_ok=True)

    dataset_name = config.data.active_datasets[0] if config.data.active_datasets else "unknown"
    run_name = get_run_name(
        config.feature_extraction.model_name,
        config.feature_extraction.segment_duration
    )
    datamodule = MusicDataModule(config)
    model = GenreClassificationModule(config)
    logger = TensorBoardLogger(
        save_dir=config.output.tensorboard_dir,
        name=f"{dataset_name}",
        version=run_name
    )
    callbacks = setup_callbacks(config)

    ckpt_config = config.training.checkpoint
    checkpoint_dir = Path(ckpt_config.save_dir)/dataset_name/run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dynamic_filename = f"{dataset_name}-best-" + ckpt_config.filename

    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=dynamic_filename,
            monitor=ckpt_config.monitor,
            mode=ckpt_config.mode,
            save_top_k=ckpt_config.save_top_k,
            save_last=ckpt_config.save_last,
            verbose=False,
            auto_insert_metric_name=False
        )
    )

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config.training.log_every_n_steps,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=False # 是否打印模型结构表
    )

    print("Start training...")
    print("=" * 70)
    trainer.fit(model, datamodule=datamodule)
    print("=" * 70)
    print('\n'+"Start testing...")
    trainer.test(model, datamodule=datamodule, ckpt_path='best', verbose=False)
    print("=" * 60)
    results=model.test_results
    print(f"    Test Accuracy:  {results['acc']:.4f}")
    print(f"    Test F1 Score:  {results['f1']:.4f}")
    print(f"    Test AUROC:     {results['auroc']:.4f}")
    print("=" * 60)
    print('\n'+"Val results...")
    print("=" * 70)
    print(f"   Val Accuracy: {model.best_val_acc:.4f}")
    print(f"   Val F1 Score: {model.best_val_f1:.4f}")
    print("" + "=" * 70 + "")

if __name__ == "__main__":
    main()