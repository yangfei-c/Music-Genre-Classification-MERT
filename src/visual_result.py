import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from itertools import cycle
import pandas as pd

# 设置风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 尝试支持中文，如果乱码会回退
plt.rcParams['axes.unicode_minus'] = False


def load_pt(result_dir):
    result_path = Path(result_dir)
    pred_files = list(result_path.rglob('test_predictions.pt'))

    results = []
    for p in pred_files:
        try:
            run_name = p.parent.name
            dataset_name = p.parent.parent.name
            data = torch.load(p, map_location='cpu',weights_only=False)
            results.append({
                'dataset': dataset_name,
                'model': run_name,
                'data': data,
                'path': p.parent
            })
            print(f"✅ Loaded: {dataset_name} - {run_name}")
        except Exception as e:
            print(f"❌ Error loading {p}: {e}")
    return results


def plot_confusion_matrix(y_true, y_pred_classes, classes, title, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred_classes)
    # 归一化
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path / f'confusion_matrix_{title}.png', dpi=300)
    plt.close()


def plot_multiclass_roc(y_true, y_probs, classes, title, save_path):
    """绘制多分类ROC曲线"""
    n_classes = len(classes)

    # 将标签转换为one-hot
    y_true_one_hot = np.eye(n_classes)[y_true]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微平均ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_one_hot.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-average ROC (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path / f'roc_curve_{title}.png', dpi=300)
    plt.close()


def plot_metrics_comparison(results, save_dir):
    records = []
    for res in results:
        metrics = res['data']['metrics']
        records.append({
            'Dataset': res['dataset'],
            'Model': res['model'],
            'Accuracy': metrics['acc'],
            'F1 Score': metrics['f1'],
            'AUROC': metrics['auroc']
        })

    if not records:
        return

    df = pd.DataFrame(records)

    df_melted = df.melt(id_vars=['Dataset', 'Model'],
                        value_vars=['Accuracy', 'F1 Score', 'AUROC'],
                        var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x='Dataset', y='Score', hue='Model', errorbar=None, palette='viridis')
    plt.title('比较使用不同mert参数提取特征后接MLP进行分类的效果')
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'mert_95mVSmert_330m.png', dpi=300)
    plt.close()
    print(f"✅ Comparison plot saved to {save_dir}")


def main():
    # 假设结果在当前目录的上级 results 目录，或者是当前目录的 results
    # 根据你的配置文件，它在 '../results'，但在 plot_results.py 同级运行时可能是 './results'
    # 请根据实际情况修改下面的路径
    base_result_dir = './results'
    if not os.path.exists(base_result_dir):
        # 尝试找上级目录
        base_result_dir = '../results'

    if not os.path.exists(base_result_dir):
        print(f"❌ Result directory not found at {base_result_dir} or ./results")
        return

    print(f"📂 Scanning results in: {base_result_dir}")
    experiments = load_pt(base_result_dir)

    if not experiments:
        print("⚠️ No prediction files found.")
        return

    # 为每个实验绘制详细图表
    for exp in experiments:
        data = exp['data']
        probs = data['probs'].numpy()
        labels = data['labels'].numpy()
        classes = data['classes']
        save_path = exp['path']

        # 获取预测类别
        preds = np.argmax(probs, axis=1)

        print(f"🎨 Plotting for {exp['dataset']} - {exp['model']}...")

        # 1. 混淆矩阵
        plot_confusion_matrix(labels, preds, classes, f"{exp['dataset']}_{exp['model']}", save_path)

        # 2. ROC 曲线
        plot_multiclass_roc(labels, probs, classes, f"{exp['dataset']}_{exp['model']}", save_path)

    # 3. 汇总对比图
    # 将对比图保存在 results 根目录
    plot_metrics_comparison(experiments, base_result_dir)


if __name__ == "__main__":
    main()