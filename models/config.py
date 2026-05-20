"""
Model Configuration

Contains all hyperparameters and settings for training and inference.
"""

# =============================================================================
# ROUTING CATEGORIES
# =============================================================================

LABELS = [
    "renewal",
    "sicknote",
    "office", 
    "nurse",
    "doctor-not-urgent",
    "doctor-urgent"
]

LABEL_TO_TOKEN = {
    "renewal": "RX",
    "sicknote": "ILL",
    "office": "ADMIN",
    "nurse": "RN",
    "doctor-not-urgent": "NORMAL",
    "doctor-urgent": "STAT",
}

TOKEN_TO_LABEL = {v: k for k, v in LABEL_TO_TOKEN.items()}

# Hebrew category names for explanations
LABEL_HEBREW = {
    "renewal": "חידוש מרשם",
    "sicknote": "אישור מחלה",
    "office": "משרדי",
    "nurse": "אחות",
    "doctor-not-urgent": "רופא לא דחוף",
    "doctor-urgent": "רופא דחוף",
}

# =============================================================================
# ROUTING AGENT CONFIGURATION
# =============================================================================

ROUTING_MODEL_CONFIG = {
    # Base model
    "base_model": "meta-llama/Llama-3.2-3B-Instruct",
    
    # LoRA configuration
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    
    # Training hyperparameters
    "num_epochs": 3,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 16,  # 4 * 4
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_seq_length": 2048,
    
    # Precision
    "fp16": True,
    "bf16": False,  # Set to True if GPU supports it
    
    # Random seed
    "seed": 42,
}

# =============================================================================
# EXPLANATION AGENT CONFIGURATION
# =============================================================================

EXPLANATION_MODEL_CONFIG = {
    # Model (inference only, no fine-tuning)
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    
    # Generation parameters
    "max_new_tokens": 150,
    "temperature": 0.4,
    "top_p": 0.9,
    "do_sample": True,
}

# =============================================================================
# THRESHOLD CONFIGURATION
# =============================================================================

THRESHOLD_CONFIG = {
    # Doctor-Urgent threshold (optimized on validation set)
    "du_threshold": 0.22,
    
    # Threshold search parameters
    "threshold_candidates": [i/100 for i in range(1, 100)],  # 0.01 to 0.99
    "min_precision": 0.20,  # Minimum precision constraint
    
    # Optimization objective
    "objective": "maximize_recall_with_precision_floor",
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

EVAL_CONFIG = {
    # Metrics
    "metrics": ["macro_f1", "macro_auc", "per_class_f1", "per_class_auc"],
    
    # Safety-critical class
    "safety_class": "doctor-urgent",
    
    # Report metrics for safety class
    "safety_metrics": ["recall", "precision", "f1"],
}

# =============================================================================
# BASELINE MODELS
# =============================================================================

BASELINE_MODELS = {
    "word2vec": {
        "vector_size": 100,
        "window": 5,
        "min_count": 1,
        "classifier": "logistic_regression",
    },
    "mbert": {
        "model_name": "bert-base-multilingual-cased",
        "max_length": 256,
        "finetune_epochs": 3,
    },
    "llm_zeroshot": [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
}
