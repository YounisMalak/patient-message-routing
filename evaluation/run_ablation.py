"""
Ablation Study Script

Evaluates contribution of each prompt component.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import gc
from tqdm import tqdm
import argparse

import sys
sys.path.append('..')
from models.config import LABELS, LABEL_TO_TOKEN, THRESHOLD_CONFIG
from metrics import compute_metrics, print_metrics_table


# =============================================================================
# PROMPT COMPONENTS
# =============================================================================

# I: Instructions only
INSTRUCTIONS_ONLY = """סווג הודעות מטופלים לאחת מהקטגוריות: RX, ILL, ADMIN, RN, NORMAL, STAT.
ענה במילה אחת בלבד."""

# P: Priority rules
PRIORITY_RULES = """
## כללי עדיפות:
1. בקשה לחידוש מרשם (גם עם תסמינים) -> RX
2. בקשה לאישור מחלה (גם עם תסמינים) -> ILL
3. בקשה למרשם חדש עם תסמינים -> NORMAL
4. תסמינים דחופים בלבד (ללא בקשה אחרת) -> STAT
5. שירותי אחות (זריקות, מדידות) -> RN
"""

# K: Keywords
KEYWORDS = """
## מילות מפתח לכל קטגוריה:
- RX: חידוש מרשמים, tab, cap, box, 5mg, 30mg, 25mg, 20mg, 10mg, 100mg
- ILL: אישור מחלה, חופשת מחלה, ימי מחלה, מכתב מחלה
- ADMIN: החזר, התחייבות, טופס 17, חשבונית, טלפון, לחו"ל
- RN: להתחסן, בדיקת לחץ דם, טטנוס, טיפול פצע, זריקה
- NORMAL: הפניה, הפניה MRI, הפניה לאולטרסאונד, בדיקות דם
- STAT: כאבים בחזה, קוצר נשימה, חום גבוה, דימום, חבלה בראש
"""

# C: Category definitions
CATEGORIES = """
## קטגוריות:
### RX (renewal) - חידוש מרשם
בקשות לחידוש מרשמים לתרופות קיימות.

### ILL (sicknote) - אישור מחלה
בקשות לאישורי מחלה לעבודה או ללימודים.

### ADMIN (office) - משרדי
בקשות אדמיניסטרטיביות: תורים, טפסים, תוצאות בדיקות, מסמכים.

### RN (nurse) - אחות
בקשות לשירותי אחות: זריקות, חיסונים, מדידות, החלפת תחבושות.

### NORMAL (doctor-not-urgent) - רופא לא דחוף
ייעוץ רפואי לא דחוף, הפניות, התייעצות על תסמינים לא חריפים.

### STAT (doctor-urgent) - רופא דחוף
מצבים הדורשים טיפול מיידי: כאב חזה, קוצר נשימה, חום גבוה מאוד, דימום חמור.
"""

# F: Few-shot examples
EXAMPLES = """
## דוגמאות:
הודעה: "לחדש מרשם לכדור לחץ דם" -> תשובה: RX
הודעה: "צריכה אישור מחלה לעבודה" -> תשובה: ILL
הודעה: "מתי יש תור פנוי" -> תשובה: ADMIN
הודעה: "צריך זריקת B12" -> תשובה: RN
הודעה: "יש לי שיעול צריך מרשם לסירופ" -> תשובה: NORMAL
הודעה: "כאבים חזקים בחזה וקוצר נשימה" -> תשובה: STAT
הודעה: "הילד נפל על הראש והקיא" -> תשובה: STAT
"""

# Define ablation configurations
ABLATION_CONFIGS = {
    "I": INSTRUCTIONS_ONLY,
    "I+P": INSTRUCTIONS_ONLY + PRIORITY_RULES + "\nענה במילה אחת בלבד: RX / ILL / ADMIN / RN / NORMAL / STAT",
    "I+K": INSTRUCTIONS_ONLY + KEYWORDS + "\nענה במילה אחת בלבד: RX / ILL / ADMIN / RN / NORMAL / STAT",
    "I+C": INSTRUCTIONS_ONLY + CATEGORIES + "\nענה במילה אחת בלבד: RX / ILL / ADMIN / RN / NORMAL / STAT",
    "I+C+K+P": INSTRUCTIONS_ONLY + CATEGORIES + KEYWORDS + PRIORITY_RULES + "\nענה במילה אחת בלבד: RX / ILL / ADMIN / RN / NORMAL / STAT",
    "I+C+K+P+F": INSTRUCTIONS_ONLY + CATEGORIES + KEYWORDS + PRIORITY_RULES + EXAMPLES + "\nענה במילה אחת בלבד: RX / ILL / ADMIN / RN / NORMAL / STAT",
}


def evaluate_config(test_df, model_dir, system_prompt, config_name, du_threshold):
    """Evaluate a single prompt configuration."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    print(f"\nEvaluating: {config_name}...")
    print(f"  Prompt length: {len(system_prompt)} chars")
    
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
    
    y_probs = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  {config_name}"):
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
        
        y_probs.append(probs)
    
    # Cleanup
    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Compute metrics with fixed threshold
    y_probs = np.array(y_probs)
    results = compute_metrics(test_df['label'].values, y_probs, du_threshold)
    
    return results


def run_ablation_study(
    test_path: str,
    model_dir: str,
    output_path: str,
    du_threshold: float = None,
):
    """Run complete ablation study."""
    if du_threshold is None:
        du_threshold = THRESHOLD_CONFIG["du_threshold"]
    
    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    print(f"Fixed DU threshold: {du_threshold}")
    
    # Load test data
    test_df = pd.read_json(test_path)
    print(f"Test set: {len(test_df)} samples")
    
    all_results = {}
    
    for config_name, prompt in ABLATION_CONFIGS.items():
        results = evaluate_config(test_df, model_dir, prompt, config_name, du_threshold)
        all_results[config_name] = results
    
    # Print summary table
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Config':<15} {'Macro F1':<12} {'Macro AUC':<12} {'DU Recall':<12}")
    print("-" * 55)
    
    for config_name in ABLATION_CONFIGS.keys():
        r = all_results[config_name]
        print(f"{config_name:<15} {r['macro_f1']:<12.3f} {r['macro_auc']:<12.3f} {r['du_recall']*100:<11.1f}%")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--output", type=str, default="ablation_results.json")
    parser.add_argument("--du_threshold", type=float, default=0.22,
                        help="Fixed DU threshold")
    
    args = parser.parse_args()
    
    run_ablation_study(
        args.test_data,
        args.model_dir,
        args.output,
        args.du_threshold
    )
