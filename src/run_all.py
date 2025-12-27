import os

models = ["mert_95m", "mert_330m"]
datasets = ["gtzan", "fma_small"]

def run_command(command):
    print(f"\n🚀 Running: {command}")
    # 使用 shell=True 允许在 Windows 下运行
    ret = os.system(command)
    if ret != 0:
        print(f"❌ Failed: {command}")
        exit(1)

print("\n"+"="*60)
print("🔥Training")
print("="*60)
for model in models:
    for dataset in datasets:
        cmd = f"python train.py model={model} data.active_datasets=[{dataset}]"
        run_command(cmd)

print("\n✅ All experiments completed!")