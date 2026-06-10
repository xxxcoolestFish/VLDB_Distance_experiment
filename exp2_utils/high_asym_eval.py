"""
高非对称子集评估 (Asymmetric Stratified Evaluation) — 策略二

核心思想:
    全局 MRE 被长距离对称 pair 主导, 淹没了 L̃₁ 的方向性优势。
    应该单独评估 d(u→v) 与 d(v→u) 差异大的 pair ——
    这些才是"有向图的方向性"真正发挥作用的地方。

用法:
    from exp2_utils.high_asym_eval import load_dataset_with_reverse, \
        evaluate_with_asym

    # 1. 加载包含反向距离的数据
    test_dataset_with_rev = load_dataset_with_reverse(query_dir, ...)

    # 2. 模型评估后计算 High-Asym MRE
    metrics = evaluate_with_asym(model, test_dataloader_with_rev, ...)
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_utils import read_query_file


class AsymWorkloadDataset(torch.utils.data.Dataset):
    """带反向距离的数据集: 每样本 (u, v, d_uv, d_vu)"""

    def __init__(self, queries, replicate=False, target_size=1_000_000):
        if replicate:
            num_copies = max(1, target_size // len(queries))
            queries = queries * num_copies
        self.queries = np.array(queries, dtype=object)
        self.D = self.queries[:, 2].astype(np.float32)
        self.D_rev = self.queries[:, 3].astype(np.float32)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        u, v, d_uv, d_vu = (self.queries[idx, 0], self.queries[idx, 1],
                             self.queries[idx, 2], self.queries[idx, 3])
        return np.int32(u), np.int32(v), np.float32(d_uv), np.float32(d_vu)


def load_dataset_with_reverse(query_dir, batch_size_test=2**20,
                               seed=42, num_workers=0, **kwargs):
    """加载包含反向距离 (d_vu) 的测试数据集。"""
    import glob
    file_names = glob.glob(os.path.join(query_dir, "*.queries"))
    file_names = [os.path.normpath(fn) for fn in file_names]
    assert len(file_names) > 0, f"No .queries files in {query_dir}"

    if len(file_names) == 2:
        test_file = None
        for f in file_names:
            if f.endswith("_test.queries"):
                test_file = f
        assert test_file is not None, "Expected *_test.queries file"
        test_data = read_query_file(test_file, **kwargs)
    elif len(file_names) == 1:
        from sklearn.model_selection import train_test_split
        full_data = read_query_file(file_names[0], **kwargs)
        _, test_data = train_test_split(full_data, test_size=0.2,
                                         random_state=seed)

    print(f"Loaded {len(test_data)} test queries with reverse distances.")

    # 过滤: 只保留有 d_vu 的 (4 列)
    test_data_4col = [q for q in test_data if len(q) == 4]
    if len(test_data_4col) < len(test_data):
        print(f"  Warning: {len(test_data) - len(test_data_4col)} queries "
              f"missing reverse distance (filtered out).")

    test_dataset = AsymWorkloadDataset(test_data_4col, replicate=True,
                                        target_size=batch_size_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test,
                                  shuffle=False, num_workers=num_workers,
                                  pin_memory=True)
    return test_dataloader


def compute_high_asym_metrics(predictions, targets, targets_rev):
    """计算全局 MRE 和高非对称 MRE。

    Args:
        predictions: np.array, shape (N,) — 预测的 d(u→v)
        targets:     np.array, shape (N,) — 真实的 d(u→v)
        targets_rev: np.array, shape (N,) — 真实的 d(v→u)

    Returns:
        dict with global_mre, high_asym_mre, asym_ratios, etc.
    """
    eps = 1e-6
    n = len(predictions)

    # 非对称比: max(d_uv, d_vu) / min(d_uv, d_vu)
    d_max = np.maximum(targets, targets_rev)
    d_min = np.minimum(targets, targets_rev)
    asym_ratio = d_max / (d_min + eps)

    # 全局 MRE
    global_mre = np.mean(np.abs(predictions - targets) / np.maximum(targets, eps))

    # 高非对称 MRE (ratio > 1.2)
    high_mask = asym_ratio > 1.2
    n_high = high_mask.sum()
    if n_high > 0:
        high_asym_mre = np.mean(
            np.abs(predictions[high_mask] - targets[high_mask])
            / np.maximum(targets[high_mask], eps))
    else:
        high_asym_mre = 0.0

    # 中非对称 MRE (1.05 < ratio <= 1.2)
    mid_mask = (asym_ratio > 1.05) & (asym_ratio <= 1.2)
    n_mid = mid_mask.sum()
    if n_mid > 0:
        mid_asym_mre = np.mean(
            np.abs(predictions[mid_mask] - targets[mid_mask])
            / np.maximum(targets[mid_mask], eps))
    else:
        mid_asym_mre = 0.0

    # 近乎对称 MRE (ratio <= 1.05)
    sym_mask = asym_ratio <= 1.05
    n_sym = sym_mask.sum()
    if n_sym > 0:
        sym_mre = np.mean(
            np.abs(predictions[sym_mask] - targets[sym_mask])
            / np.maximum(targets[sym_mask], eps))
    else:
        sym_mre = 0.0

    # 非对称比分布
    asym_buckets = {
        "ratio_1.0_1.05": float((asym_ratio <= 1.05).sum() / n * 100),
        "ratio_1.05_1.2": float(mid_mask.sum() / n * 100),
        "ratio_1.2_2.0": float(((asym_ratio > 1.2) & (asym_ratio <= 2.0)).sum() / n * 100),
        "ratio_2.0_5.0": float(((asym_ratio > 2.0) & (asym_ratio <= 5.0)).sum() / n * 100),
        "ratio_5.0_inf": float((asym_ratio > 5.0).sum() / n * 100),
    }

    return {
        "global_mre": float(global_mre),
        "global_mre_percent": float(100 * global_mre),
        "high_asym_mre": float(high_asym_mre),
        "high_asym_mre_percent": float(100 * high_asym_mre),
        "high_asym_count": int(n_high),
        "high_asym_fraction": float(100 * n_high / n),
        "mid_asym_mre_percent": float(100 * mid_asym_mre),
        "mid_asym_fraction": float(100 * n_mid / n),
        "sym_mre_percent": float(100 * sym_mre),
        "sym_fraction": float(100 * n_sym / n),
        "asym_ratio_buckets": asym_buckets,
        "asym_ratio_mean": float(asym_ratio.mean()),
        "asym_ratio_median": float(np.median(asym_ratio)),
        "asym_ratio_max": float(asym_ratio.max()),
    }


def evaluate_with_asym(model, dataloader, max_distance=1.0, device="cuda",
                        verbose=True, raw_max_distance=None):
    """带高非对称指标的完整评估。

    用包含反向距离的 dataloader 评估模型，
    输出全局 MRE + 高非对称 MRE + 非对称比分布。

    Args:
        raw_max_distance: 原始最大距离（用于反归一化）。
                          如果模型输出被归一化到 [0,1]，需要此参数
                          将预测值恢复到原始米制。
    """
    model.eval()
    model.to(device)

    predictions, targets, targets_rev = [], [], []
    total_time = 0.0

    with torch.no_grad():
        for batch in dataloader:
            i, j, d_ij, d_vu = batch
            targets.append(d_ij.cpu().numpy())
            targets_rev.append(d_vu.cpu().numpy())

            start = time.perf_counter()
            i, j = i.to(device), j.to(device)
            outputs = model.forward(i, j)
            outputs = outputs.cpu().numpy().ravel()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            total_time += time.perf_counter() - start
            predictions.append(outputs)

    predictions = np.hstack(predictions)
    targets = np.hstack(targets)
    targets_rev = np.hstack(targets_rev)

    # 反归一化: 如果 max_distance=1.0 但数据是原始米制
    if raw_max_distance is not None and raw_max_distance > 1.0:
        predictions = predictions * raw_max_distance

    metrics = compute_high_asym_metrics(predictions, targets, targets_rev)
    query_latency_us = total_time / len(targets) * 1_000_000

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Asymmetric Stratified Evaluation")
        print(f"{'='*60}")
        print(f"  Total samples:           {len(predictions):>10,}")
        print(f"  Asym ratio mean/median:  {metrics['asym_ratio_mean']:.3f} / "
              f"{metrics['asym_ratio_median']:.3f}")
        print(f"  Asym ratio max:          {metrics['asym_ratio_max']:.1f}")
        print(f"")
        print(f"  {'Category':<20} {'MRE':>8}  {'Count':>10}  {'Fraction':>10}")
        print(f"  {'-'*50}")
        print(f"  {'Symmetric (<=1.05)':<20} {metrics['sym_mre_percent']:>7.2f}%  "
              f"{int(n_sym:=metrics['sym_fraction']*len(predictions)/100):>10,}  "
              f"{metrics['sym_fraction']:>9.1f}%")
        print(f"  {'Mid Asym (1.05-1.2)':<20} {metrics['mid_asym_mre_percent']:>7.2f}%  "
              f"{int(metrics['mid_asym_fraction']*len(predictions)/100):>10,}  "
              f"{metrics['mid_asym_fraction']:>9.1f}%")
        print(f"  {'High Asym (>1.2)':<20} {metrics['high_asym_mre_percent']:>7.2f}%  "
              f"{metrics['high_asym_count']:>10,}  "
              f"{metrics['high_asym_fraction']:>9.1f}%")
        print(f"  {'-'*50}")
        print(f"  {'GLOBAL':<20} {metrics['global_mre_percent']:>7.2f}%")
        print(f"")
        print(f"  Asym Ratio Distribution:")
        for k, v in metrics['asym_ratio_buckets'].items():
            print(f"    {k}: {v:.1f}%")
        print(f"{'='*60}")
        print(f"  Query latency: {query_latency_us:.1f} us/sample")

    return predictions, targets, targets_rev, metrics


