"""
Baseline Evaluation Script

Evaluates all baseline models on the test set.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import gc
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
import argparse

import sys
sys.path.append('..')
from models.config import LABELS, LABEL_TO_TOKEN, BASELINE_MODELS
from metrics import compute_metrics, find_optimal_du_threshold, print_metrics_table


def evaluate_word2vec(train_df, val_df, test_df, config):
    """Word2Vec + Logistic Regression baseline."""
    print("\nEvaluating: Word2Vec + Logistic Regression...")
    
    def tokenize(text):
        return text.split()
    
    train_texts = [tokenize(t) for t in train_df['text'].values]
    val_texts = [tokenize(t) for t in val_df['text'].values]
    test_texts = [tokenize(t) for t in test_df['text'].values]
    
    # Train Word2Vec
    w2v = Word2Vec(
        sentences=train_texts,
        vector_size=config['vector_size'],
        window=config['window'],
        min_count=config['min_count'],
        workers=4,
        seed=42
    )
    
    def get_doc_vector(tokens, model):
        vectors = [model.wv[w] for w in tokens if w in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
    
    X_train = np.array([get_doc_vector(t, w2v) for t in train_texts])
    X_val = np.array([get_doc_vector(t, w2v) for t in val_texts])
    X_test = np.array([get_doc_vector(t, w2v) for t in test_texts])
    
    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_df['label'].values)
    
    # Get probabilities
    val_probs = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)
    
    # Align probabilities to LABELS order
    classes = clf.classes_.tolist()
    
    def align_probs(probs):
        aligned = np.zeros((len(probs), len(LABELS)))
        for i, label in enumerate(LABELS):
            if label in classes:
                aligned[:, i] = probs[:, classes.index(label)]
        return aligned
    
    val_probs = align_probs(val_probs)
    test_probs = align_probs(test_probs)
    
    # Find optimal threshold on validation
    threshold_info = find_optimal_du_threshold(val_df['label'].values, val_probs)
    print(f"  DU Threshold: {threshold_info['du_threshold']:.2f}")
    
    # Evaluate on test
    results = compute_metrics(test_df['label'].values, test_probs, threshold_info['du_threshold'])
    
    return results


def evaluate_mbert(train_df, val_df, test_df, config, finetune=False):
    """mBERT baseline (with or without fine-tuning)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from torch.utils.data import Dataset
    
    name = "mBERT (finetuned)" if finetune else "mBERT"
    print(f"\nEvaluating: {name}...")
    
    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.label_map = {l: i for i, l in enumerate(LABELS)}
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            encoded = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_len,
                padding='max_length',
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': torch.tensor(self.label_map[self.labels[idx]])
            }
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS)
    )
    
    if finetune:
        train_dataset = TextDataset(
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            tokenizer,
            config['max_length']
        )
        
        training_args = TrainingArguments(
            output_dir='./mbert_temp',
            num_train_epochs=config['finetune_epochs'],
            per_device_train_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=500,
            save_strategy='no',
            report_to='none',
            seed=42,
        )
        
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        trainer.train()
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    def get_probs(df):
        probs_list = []
        with torch.no_grad():
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {name}"):
                inputs = tokenizer(
                    str(row['text']),
                    return_tensors='pt',
                    truncation=True,
                    max_length=config['max_length'],
                    padding='max_length'
                ).to(device)
                
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                probs_list.append(probs)
        
        return np.array(probs_list)
    
    val_probs = get_probs(val_df)
    test_probs = get_probs(test_df)
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Find optimal threshold
    threshold_info = find_optimal_du_threshold(val_df['label'].values, val_probs)
    print(f"  DU Threshold: {threshold_info['du_threshold']:.2f}")
    
    # Evaluate
    results = compute_metrics(test_df['label'].values, test_probs, threshold_info['du_threshold'])
    
    return results


def evaluate_llm_zeroshot(val_df, test_df, model_name, system_prompt):
    """Zero-shot LLM baseline."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    display_name = model_name.split('/')[-1]
    print(f"\nEvaluating: {display_name} (zero-shot)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Get token IDs for labels
    label_token_ids = {}
    for label in LABELS:
        token = LABEL_TO_TOKEN[label]
        ids = tokenizer.encode(token, add_special_tokens=False)
        label_token_ids[label] = ids[0] if ids else tokenizer.unk_token_id
    
    token_id_list = [label_token_ids[l] for l in LABELS]
    device = next(model.parameters()).device
    is_mistral = 'mistral' in model_name.lower()
    
    def get_probs(df):
        probs_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {display_name}"):
            text = str(row['text']).strip()
            
            if is_mistral:
                messages = [{"role": "user", "content": f"{system_prompt}\n\nהודעה: {text}"}]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt, return_tensors='pt', truncation=True, max_length=2048
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                last_logits = outputs.logits[0, -1, :]
                label_logits = torch.tensor([last_logits[tid].float().item() for tid in token_id_list])
                probs = torch.softmax(label_logits, dim=0).cpu().numpy()
            
            probs_list.append(probs)
        
        return np.array(probs_list)
    
    val_probs = get_probs(val_df)
    test_probs = get_probs(test_df)
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Find optimal threshold
    threshold_info = find_optimal_du_threshold(val_df['label'].values, val_probs)
    print(f"  DU Threshold: {threshold_info['du_threshold']:.2f}")
    
    # Evaluate
    results = compute_metrics(test_df['label'].values, test_probs, threshold_info['du_threshold'])
    
    return results


def evaluate_our_method(val_df, test_df, model_dir):
    """Evaluate our fine-tuned method."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    print("\nEvaluating: Our Method (fine-tuned)...")
    
    # Load system prompt
    with open(os.path.join(model_dir, 'system_prompt.txt'), 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    
    label_token_ids = {l: tokenizer.encode(LABEL_TO_TOKEN[l], add_special_tokens=False)[0] for l in LABELS}
    token_id_list = [label_token_ids[l] for l in LABELS]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_probs(df):
        probs_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Our Method"):
            text = str(row['text']).strip()
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt, return_tensors='pt', truncation=True, max_length=2048
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                last_logits = outputs.logits[0, -1, :]
                label_logits = torch.tensor([last_logits[tid].float().item() for tid in token_id_list])
                probs = torch.softmax(label_logits, dim=0).cpu().numpy()
            
            probs_list.append(probs)
        
        return np.array(probs_list)
    
    val_probs = get_probs(val_df)
    test_probs = get_probs(test_df)
    
    # Cleanup
    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Find optimal threshold
    threshold_info = find_optimal_du_threshold(val_df['label'].values, val_probs)
    print(f"  DU Threshold: {threshold_info['du_threshold']:.2f}")
    
    # Evaluate
    results = compute_metrics(test_df['label'].values, test_probs, threshold_info['du_threshold'])
    
    return results


def run_all_baselines(
    train_path: str,
    val_path: str,
    test_path: str,
    model_dir: str,
    output_path: str,
):
    """Run all baseline evaluations."""
    print("=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    
    # Load data
    train_df = pd.read_json(train_path)
    val_df = pd.read_json(val_path)
    test_df = pd.read_json(test_path)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Load zero-shot prompt
    zeroshot_prompt = """סווג את ההודעה הבאה לאחת מהקטגוריות:
- RX: חידוש מרשם
- ILL: אישור מחלה
- ADMIN: משרדי (תורים, טפסים)
- RN: שירותי אחות (זריקות, חיסונים)
- NORMAL: רופא לא דחוף
- STAT: רופא דחוף (מצב חירום)

ענה במילה אחת בלבד: RX / ILL / ADMIN / RN / NORMAL / STAT"""
    
    all_results = {}
    
    # Word2Vec
    all_results['Word2Vec'] = evaluate_word2vec(
        train_df, val_df, test_df, BASELINE_MODELS['word2vec']
    )
    
    # mBERT
    all_results['mBERT'] = evaluate_mbert(
        train_df, val_df, test_df, BASELINE_MODELS['mbert'], finetune=False
    )
    all_results['mBERT (finetuned)'] = evaluate_mbert(
        train_df, val_df, test_df, BASELINE_MODELS['mbert'], finetune=True
    )
    
    # Zero-shot LLMs
    for model_name in BASELINE_MODELS['llm_zeroshot']:
        display_name = model_name.split('/')[-1]
        try:
            all_results[display_name] = evaluate_llm_zeroshot(
                val_df, test_df, model_name, zeroshot_prompt
            )
        except Exception as e:
            print(f"Error with {display_name}: {e}")
    
    # Our method
    all_results['Our Method'] = evaluate_our_method(val_df, test_df, model_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, results in all_results.items():
        print_metrics_table(results, name)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--output", type=str, default="baseline_results.json")
    
    args = parser.parse_args()
    
    run_all_baselines(
        args.train_data,
        args.val_data,
        args.test_data,
        args.model_dir,
        args.output
    )
