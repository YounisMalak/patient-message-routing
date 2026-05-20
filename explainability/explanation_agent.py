"""
Explanation Agent Module

Generates natural language explanations for routing decisions.
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Dict

import sys
sys.path.append('..')
from models.config import EXPLANATION_MODEL_CONFIG, LABEL_HEBREW, LABELS


class ExplanationAgent:
    """
    Agent for generating explanations of routing decisions.
    
    Uses a separate LLM (Llama-3.1-8B-Instruct) to generate
    human-readable justifications for each classification.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
    ):
        """
        Initialize the explanation agent.
        
        Args:
            model_name: Model to use (default from config)
            device: Device to run on (default: auto)
        """
        self.model_name = model_name or EXPLANATION_MODEL_CONFIG["model_name"]
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading explanation model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        
        # Generation parameters
        self.gen_config = {
            "max_new_tokens": EXPLANATION_MODEL_CONFIG["max_new_tokens"],
            "temperature": EXPLANATION_MODEL_CONFIG["temperature"],
            "top_p": EXPLANATION_MODEL_CONFIG["top_p"],
            "do_sample": EXPLANATION_MODEL_CONFIG["do_sample"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
    
    def generate_explanation(
        self,
        message: str,
        category: str,
        category_token: Optional[str] = None,
    ) -> str:
        """
        Generate explanation for a routing decision.
        
        Args:
            message: Patient message text
            category: Predicted category (e.g., 'doctor-urgent')
            category_token: Token used for category (e.g., 'STAT')
            
        Returns:
            Generated explanation text
        """
        # Get Hebrew category name
        category_hebrew = LABEL_HEBREW.get(category, category)
        
        # Get category context
        category_context = self._get_category_context(category)
        
        # Build prompt
        prompt = self._build_explanation_prompt(
            message, category_hebrew, category_token, category_context
        )
        
        # Generate
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            input_text, return_tensors='pt', truncation=True, max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.gen_config)
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Clean up response
        cleaned = self._clean_explanation(response)
        
        return cleaned
    
    def _build_explanation_prompt(
        self,
        message: str,
        category_hebrew: str,
        category_token: str,
        context: str,
    ) -> str:
        """Build the explanation prompt."""
        
        prompt = f"""הסבר בקצרה מדוע ההודעה הבאה סווגה לקטגוריה "{category_hebrew}".

הודעת המטופל:
"{message}"

קטגוריה: {category_hebrew} ({category_token}) - {context}

הנחיות חשובות:
- כתוב 2-3 משפטים בלבד
- התמקד במה שהמטופל ביקש או תיאר
- ציין מילים מרכזיות מההודעה שהובילו לסיווג
- כתוב בשפה טבעית ופשוטה
- אל תזכיר "כללי עדיפות", "בסיס ידע", או מונחים טכניים
- אל תחזור על הגדרת הקטגוריה

דוגמאות להסברים טובים:
- "המטופל ביקש לחדש מרשם לתרופה EUTHYROX, לכן ההודעה סווגה כחידוש מרשם."
- "המטופל דיווח על נשיכת כלב, מצב הדורש בדיקה רפואית דחופה."
- "המטופל ביקש תור לרופא, שזו בקשה אדמיניסטרטיבית."

הסבר:"""
        
        return prompt
    
    def _get_category_context(self, category: str) -> str:
        """Get context description for category."""
        contexts = {
            "renewal": "בקשות לחידוש מרשמים לתרופות קיימות",
            "sicknote": "בקשות לאישורי מחלה לעבודה או ללימודים",
            "office": "בקשות אדמיניסטרטיביות: תורים, טפסים, תוצאות בדיקות, מסמכים",
            "nurse": "בקשות לשירותי אחות: זריקות, חיסונים, מדידות, החלפת תחבושות",
            "doctor-not-urgent": "ייעוץ רפואי לא דחוף, הפניות, התייעצות על תסמינים לא חריפים",
            "doctor-urgent": "מצבים הדורשים טיפול מיידי: כאב חזה, קוצר נשימה, חום גבוה מאוד, דימום חמור",
        }
        return contexts.get(category, "")
    
    def _clean_explanation(self, text: str) -> str:
        """Clean and format the generated explanation."""
        # Remove common artifacts
        text = text.strip()
        
        # Split on artifacts and keep first segment
        artifacts = ['###', '##', '#', '```', '{', 'http', '---']
        for artifact in artifacts:
            if artifact in text:
                text = text.split(artifact)[0].strip()
        
        # Remove technical terms
        tech_terms = ['כללי העדיפות', 'בסיס הידע', 'הקטגוריה מוגדרת']
        for term in tech_terms:
            if term in text:
                sentences = text.split('.')
                sentences = [s for s in sentences if term not in s]
                text = '.'.join(sentences)
        
        # Extract first paragraph
        paragraphs = text.split('\n\n')
        if paragraphs:
            text = paragraphs[0].strip()
        
        # Remove formatting markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)       # Italic
        text = re.sub(r'^[•\-\*]\s*', '', text)          # Bullets
        text = text.strip('"\'')
        
        # Ensure proper ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def batch_generate(self, samples: List[Dict]) -> List[str]:
        """
        Generate explanations for multiple samples.
        
        Args:
            samples: List of dicts with 'message' and 'category' keys
            
        Returns:
            List of generated explanations
        """
        explanations = []
        for sample in samples:
            explanation = self.generate_explanation(
                message=sample['message'],
                category=sample['category'],
                category_token=sample.get('category_token'),
            )
            explanations.append(explanation)
        return explanations


if __name__ == "__main__":
    # Example usage
    agent = ExplanationAgent()
    
    # Test examples
    test_cases = [
        {
            "message": "צריך לחדש מרשם לכדורי לחץ דם",
            "category": "renewal",
            "category_token": "RX",
        },
        {
            "message": "כאבים חזקים בחזה וקוצר נשימה",
            "category": "doctor-urgent",
            "category_token": "STAT",
        },
        {
            "message": "רוצה להתחסן נגד שפעת",
            "category": "nurse",
            "category_token": "RN",
        },
    ]
    
    print("Generating explanations...")
    for case in test_cases:
        explanation = agent.generate_explanation(
            case["message"], case["category"], case["category_token"]
        )
        print(f"\nMessage: {case['message']}")
        print(f"Category: {case['category']}")
        print(f"Explanation: {explanation}")
