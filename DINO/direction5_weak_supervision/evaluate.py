"""
=============================================================================
 Hướng 5: Weak Supervision — Evaluation & Benchmark Suite
 Đánh giá so sánh:
   1. Nhãn mềm của Label Model vs Majority Vote vs Dawid-Skene vs Context-Aware Markov
   2. Hiệu năng của End Model (DINOv3 + Causal GRU) trên tập Gold Standard
=============================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from direction5_weak_supervision.label_model import ContextAwareMarkovLabelModel
from direction5_weak_supervision.end_model import WeakSupervisionEndModel


def majority_vote_predict(lf_matrix: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """
    Baseline 1: Bỏ phiếu đa số (Majority Vote).
    Bỏ qua phiếu Abstain (-1). Nếu tất cả abstain, trả về phân phối đều.
    """
    T, J = lf_matrix.shape
    probs = np.zeros((T, num_classes), dtype=np.float32)

    for t in range(T):
        votes = lf_matrix[t]
        valid_votes = votes[votes >= 0]
        if len(valid_votes) == 0:
            probs[t] = 1.0 / num_classes
        else:
            counts = np.bincount(valid_votes, minlength=num_classes)
            probs[t] = counts / counts.sum()

    return probs


def dawid_skene_predict(
    lf_matrix: np.ndarray,
    num_classes: int = 4,
    max_iter: int = 20,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Baseline 2: Dawid-Skene Model cổ điển (độc lập có điều kiện, không xét quan hệ Markov thời gian).
    """
    T, J = lf_matrix.shape
    # Khởi tạo posterior bằng Majority Vote
    posterior = majority_vote_predict(lf_matrix, num_classes)
    
    # Ma trận nhầm lẫn của từng LF: (J, K, K) -> P(LF=l | Y=k)
    confusion = np.full((J, num_classes, num_classes), 0.3 / (num_classes - 1))
    for j in range(J):
        for k in range(num_classes):
            confusion[j, k, k] = 0.7

    class_priors = np.mean(posterior, axis=0)

    for _ in range(max_iter):
        # M-step: Cập nhật confusion matrix
        for j in range(J):
            for k in range(num_classes):
                denom = np.sum(posterior[:, k]) + eps
                for l in range(num_classes):
                    mask = (lf_matrix[:, j] == l)
                    numer = np.sum(posterior[mask, k]) + eps
                    confusion[j, k, l] = numer / denom
                # Chuẩn hóa hàng
                confusion[j, k] /= np.sum(confusion[j, k])

        # E-step: Cập nhật posterior
        for t in range(T):
            log_p = np.log(np.maximum(class_priors, eps))
            for j in range(J):
                obs = lf_matrix[t, j]
                if obs >= 0:
                    log_p += np.log(np.maximum(confusion[j, :, obs], eps))
            # Log-sum-exp
            m = np.max(log_p)
            p = np.exp(log_p - m)
            posterior[t] = p / np.sum(p)

    return posterior


def evaluate_label_aggregation(
    gold_labels: np.ndarray,
    predicted_probs: np.ndarray,
    method_name: str = "Method"
) -> Dict[str, float]:
    """
    Tính các chỉ số đo lường chuẩn xác: Accuracy, Macro F1, Brier Score.
    """
    pred_classes = np.argmax(predicted_probs, axis=-1)
    acc = accuracy_score(gold_labels, pred_classes)
    f1 = f1_score(gold_labels, pred_classes, average="macro", zero_division=0)
    
    # Brier score đa lớp: mean(sum((p_k - y_k)^2))
    K = predicted_probs.shape[1]
    one_hot = np.eye(K)[gold_labels]
    brier = float(np.mean(np.sum((predicted_probs - one_hot) ** 2, axis=1)))

    print(f"📊 [{method_name}]")
    print(f"   Accuracy:    {acc * 100:.2f}%")
    print(f"   Macro F1:    {f1:.4f}")
    print(f"   Brier Score: {brier:.4f}")

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "brier_score": brier,
    }


def evaluate_end_model(
    model: WeakSupervisionEndModel,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Đánh giá End Model trên tập Gold Test hoàn chỉnh.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                rgb_seq, targets = batch[0], batch[1]
            else:
                rgb_seq = batch["rgb_seq"]
                targets = batch["gold_label"]

            rgb_seq = rgb_seq.to(device)
            logits = model(rgb_seq)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            if isinstance(targets, torch.Tensor):
                all_targets.extend(targets.cpu().numpy())
            else:
                all_targets.extend(targets)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    print("\n🎯 [End Model Performance on Gold Test Set]")
    print(f"   Accuracy: {acc * 100:.2f}%")
    print(f"   Macro F1: {f1:.4f}")
    print("   Confusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    # Test mô phỏng nhanh
    np.random.seed(42)
    T = 120
    J = 5
    K = 4
    true_y = np.random.randint(0, K, size=T)
    # Giả lập LF với nhiễu và Abstain
    sim_lfs = np.zeros((T, J), dtype=int)
    for j in range(J):
        for t in range(T):
            if np.random.rand() < 0.2:
                sim_lfs[t, j] = -1 # Abstain
            elif np.random.rand() < 0.7:
                sim_lfs[t, j] = true_y[t] # Đúng
            else:
                sim_lfs[t, j] = np.random.randint(0, K) # Sai

    mv_probs = majority_vote_predict(sim_lfs, K)
    evaluate_label_aggregation(true_y, mv_probs, "Majority Vote")

    ds_probs = dawid_skene_predict(sim_lfs, K)
    evaluate_label_aggregation(true_y, ds_probs, "Dawid-Skene")
