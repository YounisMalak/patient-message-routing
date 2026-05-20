"""
Evaluation Metrics Module

Computes classification metrics including AUC, F1, precision, recall.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple

# Import from config
import sys
sys.path.append('..')
from models.config import LABELS, THRESHOLD_CONFIG


def compute_metrics(
    y_true: List[str],
    y_probs: np.ndarray,
    du_threshold: float = None,
) -> Dict:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities (n_samples x n_classes)
        du_threshold: Threshold for Doctor-Urgent (default from config)
        
    Returns:
        Dictionary with all metrics
    """
    if du_threshold is None:
        du_threshold = THRESHOLD_CONFIG["du_threshold"]
    
    y_true = list(y_true)
    y_probs = np.array(y_probs)
    
    du_idx = LABELS.index('doctor-urgent')
    
    # Generate predictions using DU threshold
    y_pred = []
    for probs in y_probs:
        if probs[du_idx] >= du_threshold:
            y_pred.append('doctor-urgent')
        else:
            y_pred.append(LABELS[np.argmax(probs)])
    
    results = {}
    
    # Overall metrics
    results['macro_f1'] = f1_score(
        y_true, y_pred, labels=LABELS, average='macro', zero_division=0
    )
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    
    results['per_class_f1'] = {label: float(f1[i]) for i, label in enumerate(LABELS)}
    results['per_class_recall'] = {label: float(recall[i]) for i, label in enumerate(LABELS)}
    results['per_class_precision'] = {label: float(precision[i]) for i, label in enumerate(LABELS)}
    
    # Doctor-Urgent specific metrics
    results['du_recall'] = float(recall[du_idx])
    results['du_precision'] = float(precision[du_idx])
    results['du_f1'] = float(f1[du_idx])
    
    # AUC (threshold-independent)
    y_true_bin = label_binarize(y_true, classes=LABELS)
    results['per_class_auc'] = {}
    
    for i, label in enumerate(LABELS):
        try:
            results['per_class_auc'][label] = float(
                roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            )
        except ValueError:
            # Handle case where only one class present
            results['per_class_auc'][label] = 0.5
    
    results['macro_auc'] = np.mean(list(results['per_class_auc'].values()))
    results['du_auc'] = results['per_class_auc']['doctor-urgent']
    
    # Store threshold used
    results['du_threshold'] = du_threshold
    
    return results


def find_optimal_du_threshold(
    y_true: List[str],
    y_probs: np.ndarray,
    min_precision: float = None,
) -> Dict:
    """
    Find optimal Doctor-Urgent threshold on validation set.
    
    Strategy: Maximize recall while maintaining minimum precision.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        min_precision: Minimum precision constraint (default from config)
        
    Returns:
        Dictionary with optimal threshold and metrics
    """
    if min_precision is None:
        min_precision = THRESHOLD_CONFIG["min_precision"]
    
    threshold_candidates = THRESHOLD_CONFIG["threshold_candidates"]
    
    y_true = list(y_true)
    y_probs = np.array(y_probs)
    du_idx = LABELS.index('doctor-urgent')
    
    best_threshold = 0.50
    best_recall = 0
    best_precision = 0
    
    for threshold in threshold_candidates:
        # Compute metrics at this threshold
        tp = fp = fn = 0
        
        for i, (true_label, probs) in enumerate(zip(y_true, y_probs)):
            predicted_du = probs[du_idx] >= threshold
            actual_du = true_label == 'doctor-urgent'
            
            if predicted_du and actual_du:
                tp += 1
            elif predicted_du and not actual_du:
                fp += 1
            elif not predicted_du and actual_du:
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Check if this threshold is better
        if precision >= min_precision and recall > best_recall:
            best_recall = recall
            best_threshold = threshold
            best_precision = precision
    
    # Fallback: if no threshold meets precision constraint, use best F1
    if best_recall == 0:
        best_f1 = 0
        for threshold in threshold_candidates:
            tp = fp = fn = 0
            
            for i, (true_label, probs) in enumerate(zip(y_true, y_probs)):
                predicted_du = probs[du_idx] >= threshold
                actual_du = true_label == 'doctor-urgent'
                
                if predicted_du and actual_du:
                    tp += 1
                elif predicted_du and not actual_du:
                    fp += 1
                elif not predicted_du and actual_du:
                    fn += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
        
        print(f"Warning: No threshold achieves {min_precision*100:.0f}% precision, using best F1")
    
    return {
        'du_threshold': best_threshold,
        'du_precision': best_precision,
        'du_recall': best_recall,
    }


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize by row (true labels)
        
    Returns:
        Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
    
    return cm


def print_metrics_table(results: Dict, model_name: str = "Model"):
    """Print metrics in a formatted table."""
    print(f"\n{'='*60}")
    print(f"Results for: {model_name}")
    print(f"{'='*60}")
    
    print(f"\nOverall Metrics:")
    print(f"  Macro F1:  {results['macro_f1']:.3f}")
    print(f"  Macro AUC: {results['macro_auc']:.3f}")
    
    print(f"\nDoctor-Urgent (Safety-Critical):")
    print(f"  Recall:    {results['du_recall']*100:.1f}%")
    print(f"  Precision: {results['du_precision']*100:.1f}%")
    print(f"  F1:        {results['du_f1']:.3f}")
    print(f"  Threshold: {results['du_threshold']:.2f}")
    
    print(f"\nPer-Class F1:")
    for label in LABELS:
        print(f"  {label:<20} {results['per_class_f1'][label]:.3f}")
    
    print(f"\nPer-Class AUC:")
    for label in LABELS:
        print(f"  {label:<20} {results['per_class_auc'][label]:.3f}")
