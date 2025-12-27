'''
提取MERT特征，30s为单位，最短不低于25秒
相关配置可在config中修改
'''
import os
import warnings
from pathlib import Path
import hydra
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, AutoModel
proxy_url = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = proxy_url
os.environ['https_proxy'] = proxy_url
from transformers import logging as transformers_logging
# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
transformers_logging.set_verbosity_error()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#
from music_genre.src.utils import get_feature_dir

class MERTFeatureExtractor:
    """MERT特征提取器"""
    def __init__(self, config: DictConfig):
        self.config = config
        feat_config = config.feature_extraction
        self.model_name = feat_config.model_name
        self.sample_rate = feat_config.sample_rate
        self.segment_duration = feat_config.segment_duration
        self.use_segmentation = feat_config.use_segmentation
        self.min_duration = feat_config.min_duration
        self.device = torch.device(feat_config.device if torch.cuda.is_available() else 'cpu')
        # 加载模型
        print(f">>Loading MERT model:")
        print(f"    {self.model_name}")
        self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True).to(self.device)
        self.model.eval()  # 设置为评估模式
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name,trust_remote_code=True)

    def resample_audio(self,waveform: torch.Tensor,orig_sr: int) -> torch.Tensor:
        '''
        orig_sr-->if_resample
        '''
        if orig_sr != self.sample_rate:
            resampler = T.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform

    def segment_audio(self,waveform: torch.Tensor) -> list[torch.Tensor]:
        '''
        audio-->if_segment
        '''
        segment_samples = int(self.segment_duration * self.sample_rate)
        min_samples = int(self.min_duration * self.sample_rate)
        total_samples = waveform.size(0)
        segments = []
        if total_samples <= segment_samples:
            if total_samples >= min_samples:
                segments.append(waveform)
        else:
            for start in range(0, total_samples, segment_samples):
                end = min(start + segment_samples, total_samples)
                segment = waveform[start:end]
                if segment.size(0) >= min_samples:
                    segments.append(segment)
        return segments

    def extract_from_segment(self,segment: torch.Tensor):
        '''
        segment-->extract_feature
        '''
        inputs = self.processor(segment.numpy(),sampling_rate=self.sample_rate,return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        all_layers = torch.stack(hidden_states[1:]).squeeze(1)  # [layers, time, dim]
        features = all_layers.mean(dim=1)  # [layers, dim]
        return features.cpu().numpy()

    def process_audio_file(self,audio_path: Path,output_dir: Path) -> bool:
        '''
        audio-->check_audio-->if_resample-->if_segment-->extract_feature
        '''
        first_segment = output_dir / "segment_0.npy"
        if first_segment.exists():
            return True
        try:
            waveform, orig_sr = torchaudio.load(str(audio_path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            waveform = waveform.squeeze()
            if waveform.numel() == 0 or torch.isnan(waveform).any():
                print(f"⚠️ Invalid audio: {audio_path.name}")
                return False
            if waveform.size(0) < orig_sr * self.min_duration:
                print(f"⚠️ Audio too short: {audio_path.name}")
                return False
            waveform = self.resample_audio(waveform, orig_sr)

            if waveform.size(0) < self.sample_rate * self.min_duration:
                return False

            if self.use_segmentation:
                segments = self.segment_audio(waveform)
                if not segments:
                    return False

                for i, segment in enumerate(segments):
                    save_path = output_dir / f"segment_{i}.npy"
                    if not save_path.exists():
                        features = self.extract_from_segment(segment)
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(str(save_path),features)
            else:
                features = self.extract_from_segment(waveform)
                first_segment.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(first_segment),features)
            return True
        except Exception as e:
            print(f"❌ Error processing {audio_path.name}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
    def process_dataset(self, dataset_name: str):
        """
        dataset-->extract_features
        """
        dataset_config = getattr(self.config.data, dataset_name)
        audio_dir = Path(dataset_config.audio_dir)
        output_dir = get_feature_dir(
            base_dir=dataset_config.feature_dir,
            model_name_full=self.config.feature_extraction.model_name,
            duration=self.config.feature_extraction.segment_duration
        )
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f">>Created feature directory: {output_dir}")

        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        # 扫描音频文件
        print(f">>Scanning {audio_dir}...")
        audio_files = []
        for ext in ['.mp3', '.wav']:
            audio_files.extend(audio_dir.rglob(f'*{ext}'))
        total = len(audio_files)
        print(f"    Found {total} audio files")
        if total == 0:
            print("⚠️ No audio files found!")
            return
        # 处理文件
        success_count = 0
        print(f">>Start extraction")
        with tqdm(total=total, desc="   Extracting features") as pbar:
            for audio_path in audio_files:
                # 构建输出路径
                relative_path = audio_path.relative_to(audio_dir)
                file_output_dir = output_dir / relative_path.stem
                # 处理
                success = self.process_audio_file(audio_path, file_output_dir)
                if success:
                    success_count += 1
                pbar.update(1)
                pbar.set_postfix({'Success': f'{success_count}/{total}'})

        print(f"\n✅ Feature extraction complete!")
        print(f"   Processed: {success_count}/{total} files")
        print(f"   Output directory: {output_dir}\n")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """主函数"""
    print("=" * 60)
    print("🎵 MERT Feature Extraction")
    print("=" * 60)
    # 创建提取器
    extractor = MERTFeatureExtractor(config)
    # 处理所有活跃数据集
    for dataset_name in config.data.active_datasets:
        print(f">>Processing dataset:")
        print(f"    {dataset_name}")
        extractor.process_dataset(dataset_name)
if __name__ == "__main__":
    main()
