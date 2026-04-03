from collections import defaultdict
from typing import Dict, List, Tuple


def precision_recall_f1_per_label(true_labels: List[str], pred_labels: List[str]) -> Dict[str, Dict[str, float]]:
    assert len(true_labels) == len(pred_labels)
    labels = sorted(set(true_labels) | set(pred_labels))

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for t, p in zip(true_labels, pred_labels):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    metrics = {}
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        metrics[label] = {"precision": p, "recall": r, "f1": f1, "tp": tp[label], "fp": fp[label], "fn": fn[label]}
    return metrics


def macro_micro_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    label_metrics = precision_recall_f1_per_label(true_labels, pred_labels)

    macro_p = sum(m["precision"] for m in label_metrics.values()) / len(label_metrics)
    macro_r = sum(m["recall"] for m in label_metrics.values()) / len(label_metrics)
    macro_f1 = sum(m["f1"] for m in label_metrics.values()) / len(label_metrics)

    # micro
    total_tp = sum(m["tp"] for m in label_metrics.values())
    total_fp = sum(m["fp"] for m in label_metrics.values())
    total_fn = sum(m["fn"] for m in label_metrics.values())

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    exact_match = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)

    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "exact_match": exact_match,
    }


def compute_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict[str, object]:
    prim = precision_recall_f1_per_label(true_labels, pred_labels)
    mm = macro_micro_metrics(true_labels, pred_labels)
    return {
        "per_label": prim,
        **mm,
        "support": len(true_labels),
    }
