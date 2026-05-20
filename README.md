# Routing Patient Messages in Primary Care Using Explainable Ontology-Guided Large Language Models

This repository contains the code accompanying our paper:

> **Routing Patient Messages in Primary Care Using Explainable Ontology-Guided Large Language Models**
> Malak Younis, Meira Levy, Dikla Agur-Cohen, Efrat Moshe, Adir Solomon

## Overview

Accurately routing patient-generated messages is a major challenge in primary care, where clinicians must manage large volumes of unstructured text. We propose a knowledge-informed decision-support method that combines large language models (LLMs) with a domain-specific clinical ontology, developed in close collaboration with pharmacists and a senior family physician.

The method has three main components:

1. **Ontology Construction** — An expert-informed, domain-specific clinical ontology that encodes categories, keywords, and priority rules for routing decisions in family medicine.
2. **Routing Agent** — A LoRA-fine-tuned `Llama-3.2-3B-Instruct` model that performs the actual classification, guided by a structured prompt derived from the ontology.
3. **Explanation Agent** — A `Llama-3.1-8B-Instruct` model (inference only) that generates concise natural-language justifications for each routing decision, supporting clinician trust and explainability.

The system is evaluated on a real-world Hebrew patient-message corpus collected from the largest healthcare provider in our region under ethical approval (Helsinki committee), and has been deployed in a real-time human-in-the-loop pilot.

## Repository Structure

```
├── data/
│   ├── prepare_data.py          # Data loading, preprocessing, splitting, oversampling
│   └── sample_data.json         # Synthetic example messages illustrating the data format
├── ontology/
│   ├── keywords.json            # Categories, keywords, and priority rules (structured)
│   └── system_prompt.txt        # Full ontology-driven routing prompt (Hebrew)
├── models/
│   ├── config.py                # Model, LoRA, threshold, and baseline configuration
│   └── train_routing_agent.py   # LoRA fine-tuning script for the routing agent
├── evaluation/
│   ├── run_baselines.py         # Full baseline benchmark (Word2Vec, mBERT, LLMs, ours)
│   ├── run_ablation.py          # Ablation over prompt components (I, P, K, C, F)
│   └── metrics.py               # AUC, F1, per-class metrics, threshold search
├── explainability/
│   ├── explanation_agent.py     # Explanation generation agent (Llama-3.1-8B)
│   └── explanation_prompt.txt   # Explanation prompt template (Hebrew)
├── utils/
│   └── helpers.py               # Seeds, I/O, device utilities
├── requirements.txt             # Python dependencies
└── README.md
```

## Routing Categories

| Category               | Token    | Description                                                                  |
| ---------------------- | -------- | ---------------------------------------------------------------------------- |
| Prescription Renewal   | `RX`     | Requests to renew existing medication prescriptions                          |
| Illness Confirmation   | `ILL`    | Requests for sick-leave certificates for work or school                      |
| Office / Administrative| `ADMIN`  | Administrative requests (appointments, forms, documents, reimbursements)     |
| Nursing                | `RN`     | Nursing services (injections, vaccinations, measurements, dressing changes)  |
| Doctor – Not Urgent    | `NORMAL` | Non-urgent medical consultations, referrals, non-acute symptoms              |
| Doctor – Urgent        | `STAT`   | Time-sensitive clinical risk (chest pain, shortness of breath, severe bleeding) |

## Data

Due to privacy constraints and healthcare regulations, the original patient messages cannot be released. The dataset was collected under ethical approval from an institutional Helsinki committee.

We provide `data/sample_data.json`, a small set of synthetic Hebrew messages (with English translations) illustrating the expected data format:

```json
{
  "id": 1,
  "text": "צריך לחדש מרשם לכדורי לחץ דם",
  "label": "renewal",
  "translation": "Need to renew prescription for blood pressure pills"
}
```

Labels follow the canonical names: `renewal`, `sicknote`, `office`, `nurse`, `doctor-not-urgent`, `doctor-urgent`.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-compatible GPU (16 GB+ VRAM recommended for fine-tuning the routing agent; 24 GB+ recommended for running the 8B explanation agent)
- All models are run **locally**; no external APIs are used.

## Installation

```bash
git clone https://github.com/YounisMalak/patient-message-routing.git
cd patient-message-routing

python -m venv venv
source venv/bin/activate          # Linux / macOS
# .\venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Usage

### 1. Prepare data splits

Stratified 60 / 20 / 20 split with oversampling applied only to the training set:

```bash
python data/prepare_data.py \
    --data_path your_data.json \
    --output_dir ./data \
    --test_size 0.2 \
    --val_size 0.2 \
    --seed 42
```

### 2. Fine-tune the routing agent

```bash
python models/train_routing_agent.py \
    --train_data data/train.json \
    --system_prompt ontology/system_prompt.txt \
    --output_dir ./finetuned_routing_agent \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

LoRA defaults: `r=16`, `alpha=32`, `dropout=0.1`, with effective batch size 16 (per-device 4 × gradient accumulation 4).

### 3. Run the baseline benchmark

```bash
python evaluation/run_baselines.py \
    --train_data data/train.json \
    --val_data   data/val.json \
    --test_data  data/test.json \
    --model_dir  ./finetuned_routing_agent \
    --output     baseline_results.json
```

### 4. Run the ablation study

```bash
python evaluation/run_ablation.py \
    --test_data  data/test.json \
    --model_dir  ./finetuned_routing_agent \
    --output     ablation_results.json \
    --du_threshold 0.22
```

### 5. Generate explanations

```python
from explainability.explanation_agent import ExplanationAgent

agent = ExplanationAgent()  # uses Llama-3.1-8B-Instruct by default
explanation = agent.generate_explanation(
    message="כאבים חזקים בחזה וקוצר נשימה",
    category="doctor-urgent",
    category_token="STAT",
)
print(explanation)
```

## Model Details

### Routing Agent

- **Base model:** `meta-llama/Llama-3.2-3B-Instruct`
- **Fine-tuning:** LoRA (Low-Rank Adaptation), `r=16`, `alpha=32`, dropout `0.1`
- **Training:** 3 epochs, effective batch size 16, learning rate `2 × 10⁻⁴`, warmup ratio 0.1
- **Max sequence length:** 2048 tokens

### Explanation Agent

- **Model:** `meta-llama/Llama-3.1-8B-Instruct` (inference only, no fine-tuning)
- **Generation:** `max_new_tokens=150`, `temperature=0.4`, `top_p=0.9`

## Prompts

### Routing prompt (English translation)

The routing agent receives the full ontology-driven prompt below as the system message. The actual prompt used at training and inference is in Hebrew and lives in [`ontology/system_prompt.txt`](ontology/system_prompt.txt).

> Classify patient messages into one of six categories.
>
> **Categories:**
>
> - **RX (renewal) — Prescription Renewal.** Requests to renew existing medication prescriptions. *Keywords:* prescription renewal, tab, cap, box, 5mg, 30mg, 25mg, 20mg, 10mg, 100mg.
> - **ILL (sicknote) — Illness Confirmation.** Requests for sick-leave certificates for work or school. *Keywords:* sick leave certificate, sick leave, illness leave, illness certificate, days of illness, medical leave letter, illness letter for work.
> - **ADMIN (office) — Office / Administrative.** Administrative requests such as appointments, forms, test results, and documents. *Keywords:* reimbursement, reimbursement for medications, authorization, financial authorization, Form 17, invoice, form, phone, abroad, account number.
> - **RN (nurse) — Nursing.** Requests for nursing services such as injections, vaccinations, measurements, and dressing changes. *Keywords:* vaccination, blood pressure measurement, diphtheria, vaccine completion, tetanus, wound care, dressing prescription, catheter prescription, HPV, DPT.
> - **NORMAL (doctor-not-urgent) — Doctor, Not Urgent.** Non-urgent medical consultations, referrals, and discussions of non-acute symptoms. *Keywords:* pediatric gastroenterology, referral, MRI referral, ECG referral, ultrasound referral, breast ultrasound referral, blood test referral, mammography referral, Prevnar vaccine, pneumonia vaccine.
> - **STAT (doctor-urgent) — Doctor, Urgent.** Conditions requiring immediate medical attention, such as chest pain, shortness of breath, very high fever, or severe bleeding. *Keywords:* head injuries, paralysis, shortness of breath, chest pain, head trauma, bite, facial swelling, eye itching, urgent, severe bleeding.
>
> **Priority Rules:**
>
> 1. Prescription renewal request (even with symptoms) → `RX`
> 2. Illness confirmation request (even with symptoms) → `ILL`
> 3. New prescription request with symptoms → `NORMAL`
> 4. Urgent symptoms only (without any other request) → `STAT`
> 5. Nursing services (injections, measurements) → `RN`
>
> **Examples:**
>
> - "blood pressure pills need prescription renewal" → `RX`
> - "have fever need illness certificate" → `ILL`
> - "is there an available appointment" → `ADMIN`
> - "need B12 injection" → `RN`
> - "have cough need syrup prescription" → `NORMAL`
> - "strong chest pain and shortness of breath" → `STAT`
> - "the child fell and hit his head and vomited" → `STAT`
>
> Respond with a single word only: `RX` / `ILL` / `ADMIN` / `RN` / `NORMAL` / `STAT`.

### Explanation prompt (English translation)

The explanation agent uses the template in [`explainability/explanation_prompt.txt`](explainability/explanation_prompt.txt). In English:

> Briefly explain why the following message was classified into the category *{category}*.
>
> **Patient message:** *{message}*
>
> **Important instructions:**
>
> - Write only 2–3 sentences.
> - Focus on what the patient requested or described.
> - Mention key words from the message that led to the classification.
> - Write in natural, simple language.
> - Do not mention "priority rules", "knowledge base", or technical terms.
> - Do not repeat the category definition.
>
> **Examples of good explanations:**
>
> - "The patient requested a renewal for the medication EUTHYROX; therefore, the message was classified as a prescription renewal."
> - "The patient reported a dog bite, which is a condition requiring urgent medical evaluation."
> - "The patient requested an appointment with a doctor, which is an administrative request."

## Full Categories and Keywords

The table below lists the representative keywords associated with each routing category. The complete machine-readable ontology — including Hebrew translations and priority rules — is in [`ontology/keywords.json`](ontology/keywords.json), and the structured prompt actually fed into the model is in [`ontology/system_prompt.txt`](ontology/system_prompt.txt) (Hebrew).

| Category | Keywords |
| --- | --- |
| **Office Requests** | Appointment Scheduling, Form 17, Commitment, Financial Commitment, Appeal Commitment, Payment, Reimbursement, Reimbursement for Medications, Receiving Reimbursement, Phone, Contact, Substitute Doctor, Account Number, Invoice, NIS, Overseas, Referral Code, Voicemail |
| **Prescription Renewal** | Prescription Renewal, To Renew, For Which Medications, Prescription, Regular Prescription, Tab, Cap, Box, 5mg, 10mg, 20mg, 25mg, 30mg, 100mg, Regular Medications, Monthly Medications |
| **Illness Confirmation** | Sick Leave Certificate, Sick Leave Certificates, Sick Leave, Sickleave, Illness, Medical Certificate |
| **Nursing Requests** | Polio, DPT, Tetanus, Diphtheria, Pertussis, HPV, Vaccine Completion, Get Vaccinated, Stoma Equipment, Prescription for Dressing, Prescription for Catheter, Pressure Wound, Wound Treatment, Blood Pressure Test |
| **Doctor – Not Urgent** | Referral, MRI Referral, Ultrasound Referral, Breast Ultrasound Referral, ECG Referral, Blood Test Referral, Mammography Referral, Psychiatric Appointment, Visit Summary, Chronic Fatigue, Prevnar Vaccine, Pneumonia Vaccine, Papilloma Vaccine, Chronic Symptom, Emotional Therapy, Colonoscopy, Pediatric Gastro |
| **Doctor – Urgent** | Urgent, Pain, Severe Pain, Sharp Pain, Intense Pain, Extreme Pain, Headache, Shortness of Breath, Severe, Eye Itchiness, Injury, Head Injury, Swelling, Bleeding, High Fever, Severe Infection, Breathing Difficulties, Chest Pressure, Rash, Post-Surgery Pain, Wisdom Tooth |

## Classification Thresholds

In medical triage, failing to identify an urgent case is more harmful than over-triaging; therefore, threshold selection prioritizes recall over precision. We tune the decision threshold only for the **Doctor–Urgent** category for each model using grid search on the validation set, over all thresholds in `0.01 – 0.99`, with the objective of maximizing Doctor–Urgent recall while maintaining at least 20% precision. The selected thresholds reflect each model's calibration characteristics:

| Model                        | Doctor–Urgent threshold |
| ---------------------------- | ----------------------- |
| Word2Vec                     | 0.04                    |
| mBERT (no fine-tuning)       | 0.01                    |
| mBERT (fine-tuned)           | 0.03                    |
| Llama-3.2-1B-Instruct        | 0.01                    |
| Llama-3.2-3B-Instruct        | 0.19                    |
| Llama-3.1-8B-Instruct        | 0.86                    |
| Mistral-7B-Instruct-v0.3     | 0.42                    |
| **Our Method**               | **0.22**                |

Our method's moderate threshold of 0.22 achieves substantially higher urgent recall (92.3%) while satisfying the precision constraint, indicating a more reliable and clinically appropriate separation of urgent from non-urgent messages. At inference time, all models follow the same single-label decision rule: if the predicted confidence for Doctor–Urgent exceeds its calibrated threshold, the message is routed as urgent; otherwise, it is assigned to the non-urgent category with the highest predicted score.

## Explainability — Per-Category Examples

The following describes the patterns the explanation agent typically produces for each routing category. Concrete generated explanations from the test set are reproduced (in Hebrew) in the supplementary material of the paper.

### Prescription Renewal
Prescription Renewal messages are often short and highly formulaic, explicitly asking to renew an existing prescription and listing medication names and dosages. A representative message asks "Which medications should be renewed?" followed by a structured list (e.g., *CIPRALEX 20mg* and *HUMIRA 40mg*). The explanation agent justifies the routing by pointing to the explicit renewal request together with the medication list as the main evidence. Another frequent pattern is a shortage statement followed by drug names (e.g., "Missing prescriptions for the following medications … please renew prescriptions"), which the agent explains as a continuation-of-care request rather than a new clinical consultation.

### Illness Confirmation
Illness Confirmation messages request formal documentation for absence from work or school, typically anchored by explicit sick-leave phrasing and specific dates. For example, one message requests a sick-leave certificate for the same day and provides brief justification (e.g., weakness and abdominal pain). In line with the category definition and priority rules, the explanation agent treats symptom mentions as supporting context for the documentation request rather than as a request for diagnosis or treatment. Longer retroactive-leave requests (e.g., back pain over a specified rest period) follow the same logic: the explanation emphasizes the request for an illness certificate and the time span, which are decisive for routing.

### Office Requests
Office Requests are characterized by logistical, procedural, or financial intent and often contain no clinical content. Common examples include providing a phone number, asking how to schedule an appointment, requesting reimbursement for a specialist consultation, or asking for an administrative authorization for a hospital visit. The explanation agent justifies these routings by explicitly referencing administrative cues such as reimbursement, authorization, scheduling, and contact details, and by noting the lack of symptoms, medications, or treatment intent. This demonstrates that the agent relies on both positive administrative markers and negative clinical evidence.

### Nursing Requests
Nursing Requests primarily involve vaccinations and routine nursing procedures. For example, a message stating "I want to get a flu vaccine" is routed to Nursing Requests because the requested action is a standard nursing service in primary care. A recurring subtype concerns vaccine-completion questions, such as asking whether a child needs a polio booster or whether vaccination is permitted given underlying conditions. The explanation agent justifies these decisions by highlighting vaccination-related intent (e.g., vaccine, booster, completion) rather than treating them as physician consultations, aligning with the role-specific responsibilities encoded in the ontology.

### Doctor – Not Urgent
Doctor – Not Urgent messages require physician input but do not suggest immediate risk, and typically include referrals, interpretation questions, or non-acute symptom descriptions. A representative example requests a referral to a bariatric specialist following a surgeon's recommendation; the explanation agent justifies the routing by pointing to the explicit referral intent and the planned follow-up nature of the request. Additional examples include chronic complaints over months with a request for referral to a specialist, or questions about laboratory results and medication dosing. In these cases, the explanation emphasizes medical purpose (e.g., referral, results interpretation) while implicitly relying on the absence of acute danger symptoms to avoid escalation to urgent routing.

### Doctor – Urgent
Doctor – Urgent messages contain time-sensitive clinical symptoms that warrant immediate attention. A particularly informative borderline case requests a sick day for a specific date but reports chest pressure during work and notes that the patient was referred to the emergency department. The explanation agent prioritizes the high-risk symptom description (chest pressure) and the emergency referral over the administrative outcome requested, and therefore routes the message as Doctor – Urgent. This reflects a safety-oriented explainability pattern: when red-flag symptoms are present, the generated explanation explicitly foregrounds those phrases as the decisive evidence, reducing the chance of missing potentially critical cases.

## Ethical Considerations

- All experiments were conducted under ethical approval from an institutional Helsinki committee (approval number `1COM0148-22`).
- Patient data was anonymized and handled according to healthcare regulations.
- All models are run locally; no patient data leaves the local environment, and no external APIs or cloud services are used at inference time.
- The system is designed as **decision support**, not autonomous triage. Human-in-the-loop oversight by a physician or nurse is required for clinical deployment.

## GenAI Usage Disclosure

Generative AI tools were used to support language editing, phrasing refinement, and code-organization assistance during development of this codebase. All scientific content, methodological decisions, experimental design, data analysis, and reported results were produced, verified, and approved by the authors. No confidential patient data was shared with any external generative AI service. Generative AI models were also used as part of the research method (the routing and explanation agents), strictly within the local experimental framework described in the paper. The authors take full responsibility for the correctness of the code and results.

## License

This code is released under the MIT License. See [`LICENSE`](LICENSE) for details.

## Citation

If you use this code or the ideas in this work, please cite the accompanying paper (citation details will be added upon publication).
