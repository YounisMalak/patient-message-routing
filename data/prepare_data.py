"""
Data Preparation Module

Handles data loading, preprocessing, splitting, and augmentation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import json

# Routing categories
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


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from CSV or JSON file.
    
    Expected columns:
    - text (or Text): Patient message
    - label (or Label): Routing category
    
    Args:
        data_path: Path to data file
        
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    if data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        df = pd.read_csv(data_path)
    
    # Normalize column names
    df.columns = df.columns.str.lower()
    
    # Ensure required columns exist
    assert 'text' in df.columns, "Data must contain 'text' column"
    assert 'label' in df.columns, "Data must contain 'label' column"
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data.
    
    Steps:
    1. Remove empty texts
    2. Strip whitespace
    3. Filter valid labels
    4. Remove duplicates
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    # Clean text
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip()
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0]
    
    # Filter valid labels
    df = df[df['label'].isin(LABELS)]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['text'], keep='first')
    
    return df.reset_index(drop=True)


def prepare_dataset(
    data_path: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    apply_oversampling: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare train/validation/test splits.
    
    Uses stratified sampling to preserve class distribution.
    Oversampling is applied ONLY to training set.
    
    Args:
        data_path: Path to data file
        test_size: Proportion of test set (default: 0.2)
        val_size: Proportion of validation set (default: 0.2)
        random_state: Random seed for reproducibility
        apply_oversampling: Whether to oversample minority classes in training
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Load and preprocess
    df = load_data(data_path)
    df = preprocess_data(df)
    
    print(f"Total samples after preprocessing: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # First split: 80% train+val, 20% test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: train and validation from train_val
    # val_size of 0.2 overall means 0.25 of the 80%
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=train_val_df['label'],
        random_state=random_state
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Apply oversampling to training set only
    if apply_oversampling:
        train_df = oversample_minority_classes(train_df, random_state=random_state)
        print(f"\nAfter oversampling:")
        print(f"  Train: {len(train_df)}")
        print(f"  Class distribution:\n{train_df['label'].value_counts()}")
    
    # Verify no data leakage
    train_texts = set(train_df['text'])
    val_texts = set(val_df['text'])
    test_texts = set(test_df['text'])
    
    assert len(train_texts & val_texts) == 0, "Data leakage: train/val overlap!"
    assert len(train_texts & test_texts) == 0, "Data leakage: train/test overlap!"
    assert len(val_texts & test_texts) == 0, "Data leakage: val/test overlap!"
    print("\n✓ No data leakage detected")
    
    return train_df, val_df, test_df


def oversample_minority_classes(
    df: pd.DataFrame,
    target_ratio: float = 0.067,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Oversample minority classes to improve balance.
    
    Target ratio is relative to the majority class.
    Only classes below target are oversampled.
    
    Args:
        df: Training DataFrame
        target_ratio: Minimum ratio relative to majority class
        random_state: Random seed
        
    Returns:
        Oversampled DataFrame
    """
    np.random.seed(random_state)
    
    class_counts = df['label'].value_counts()
    majority_count = class_counts.max()
    target_count = int(majority_count * target_ratio)
    
    # Ensure minimum of ~200 samples for small classes
    target_count = max(target_count, 199)
    
    oversampled_dfs = [df]
    
    for label, count in class_counts.items():
        if count < target_count:
            # Number of samples to add
            n_samples = target_count - count
            
            # Sample with replacement
            label_df = df[df['label'] == label]
            sampled = label_df.sample(n=n_samples, replace=True, random_state=random_state)
            oversampled_dfs.append(sampled)
            
            print(f"  Oversampled {label}: {count} -> {target_count} (+{n_samples})")
    
    return pd.concat(oversampled_dfs, ignore_index=True)


def add_curated_examples(
    df: pd.DataFrame,
    curated_path: str
) -> pd.DataFrame:
    """
    Add expert-curated examples to the dataset.
    
    Curated examples are synthetic/manually created by domain experts
    to improve coverage of underrepresented categories.
    
    Args:
        df: Original DataFrame
        curated_path: Path to JSON file with curated examples
        
    Returns:
        DataFrame with curated examples added
    """
    with open(curated_path, 'r', encoding='utf-8') as f:
        curated = json.load(f)
    
    new_rows = []
    for label, examples in curated.items():
        for text in examples:
            new_rows.append({'text': text, 'label': label})
    
    curated_df = pd.DataFrame(new_rows)
    
    print(f"Adding {len(curated_df)} curated examples:")
    print(curated_df['label'].value_counts())
    
    return pd.concat([df, curated_df], ignore_index=True)


def format_for_training(
    df: pd.DataFrame,
    system_prompt: str
) -> List[Dict]:
    """
    Format data for LLM fine-tuning.
    
    Creates chat-format examples with system prompt.
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        system_prompt: System prompt with ontology
        
    Returns:
        List of training examples in chat format
    """
    examples = []
    
    for _, row in df.iterrows():
        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row['text']},
                {"role": "assistant", "content": LABEL_TO_TOKEN[row['label']]}
            ]
        }
        examples.append(example)
    
    return examples


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str
):
    """Save data splits to files."""
    train_df.to_json(f"{output_dir}/train.json", orient='records', force_ascii=False, indent=2)
    val_df.to_json(f"{output_dir}/val.json", orient='records', force_ascii=False, indent=2)
    test_df.to_json(f"{output_dir}/test.json", orient='records', force_ascii=False, indent=2)
    print(f"Saved splits to {output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw data")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation set proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train_df, val_df, test_df = prepare_dataset(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    save_splits(train_df, val_df, test_df, args.output_dir)
