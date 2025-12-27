from pathlib import Path

def get_run_name(model_name_full: str, duration: int) -> str:
    model_size = model_name_full.split('-')[-1]
    return f"MERT_{model_size}_{duration}s"

def get_feature_dir(base_dir, model_name_full, duration):
    base_dir = Path(base_dir)
    sub_folder_name = get_run_name(model_name_full, duration)
    return base_dir / sub_folder_name