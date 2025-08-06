# AURA
AURA: Affordance-Understanding and Risk-aware Alignment Technique for Large Language Models

## 📂 Directory Overview

- `training/`: Contains scripts to preprocess datasets and train the AURA model using Process Reward Modeling (PRM).
- `evaluation/`: Scripts for evaluating step-level **Coherence** and **Safety** classification performance.

---

## 🏗️ Training Pipeline

### 1. Prepare Your Annotated Dataset

Your input dataset should be a list of JSON objects, each structured like below:

```json
{
  "situation": "In the sprawling, half-finished expanse of the construction site, shafts of light stream in through makeshift windows ...",
  "query": "What happens if someone trips on an extension cord while carrying a heavy load?",
  "reasoning_steps": [
    "1. The construction site is cluttered with equipment and supplies, including an extension cord on the floor.",
    "2. An extension cord can present a tripping hazard if someone is not paying attention to where they step.",
    "3. Carrying a heavy load can obstruct a person’s view, making them less likely to notice hazards such as an extension cord.",
    "4. If someone trips on an extension cord while carrying a heavy load, they may lose their balance and fall.",
    "5. Falling while carrying a heavy load increases the risk of injury to the person due to the added weight and impact.",
    "6. The action of tripping and falling could also result in the heavy load being dropped, potentially causing damage to the object or to the surroundings.",
    "7. On a construction site, such an incident could further lead to a chain reaction, disturbing other workers or equipment, increasing the overall risk of accidents."
  ],
  "Coherence": [1, 1, 1, 1, 1, 1, 1],
  "Safety": [0, 1, 0, 1, 1, 1, 1]
}
```

### 2. Preprocess the Dataset into PRM Format
Before training, convert your annotated data into a Process Reward Model (PRM) format. Use a script similar to the following:
```
GOOD_TOKEN = '<+>'
BAD_TOKEN = '<->'
SEPARATOR_TOKEN = '<extra>'

SYSTEM_PROMPT = (
    "You are a reasoning critic model that evaluates a sequence of human activities step by step. "
    "Given a situation, query and a response, evaluate the logical coherence of the current step "
    "and whether it poses any safety concern, and whether any interventive warning is provided in the current step"
)

for item in annotated_data:
    situation = item["situation"]
    query = item["query"]
    max_steps = len(item["reasoning_steps"])

    for step_number in range(max_steps):
        prev_steps = item["reasoning_steps"][:step_number]
        cur_step = item["reasoning_steps"][step_number]

        # Labels
        coh = item["Coherence"][step_number]
        saf = item["Safety"][step_number]

        # Format input and label strings
        steps_str = '\n\n'.join(prev_steps)
        user_prompt = f"\n\nSituation: {situation}\n\nQuery: {query}\n\n{steps_str}"

        content_lab = f"Current Step: {cur_step}. Coherence: {GOOD_TOKEN if coh else BAD_TOKEN}, Safety: {GOOD_TOKEN if saf == 0 else BAD_TOKEN}"
        content_inp = f"Current Step: {cur_step}. Coherence: {SEPARATOR_TOKEN}, Safety: {SEPARATOR_TOKEN}"

        sample = {
            'labels': [
                {"role": "user", "content": SYSTEM_PROMPT + user_prompt},
                {"role": "assistant", "content": content_lab}
            ],
            'inputs': [
                {"role": "user", "content": SYSTEM_PROMPT + user_prompt},
                {"role": "assistant", "content": content_inp}
            ],
            'step_position': step_number
        }
        # Save or append sample to PRM dataset

```
Save the processed dataset in Arrow format.

### 3. Run Training Scripts
* Step 1: Prepare dataset
```
python training/prepare_train_dataset.py
```

* Step 2: Update tokenizer
```
python training/prepare_model.py
```

* Step 3: Start training the AffordRanker
```
bash training/train_afford_ranker.sh
```

### 📈 Evaluation Pipeline
To reproduce step-level label prediction (Coherence, Safety):
```
python evaluation/step_label_classification.py
```
This script uses the trained model to evaluate how well it predicts Coherence and Safety scores at each step of reasoning, as aligned with the gold annotations.

### 📁 Repository Structure

```
AURA/
├── training/
│   ├── prepare_train_dataset.py
│   ├── prepare_model.py
│   ├── train_afford_ranker.sh
│   └── ...
├── evaluation/
│   ├── step_label_classification.py
│   └── ...
├── README.md
└── ...

```

### 📝 Notes
* Ensure your reasoning_steps, Coherence, and Safety arrays are aligned in length.

* You may adjust the PRM prompt and token markers if needed for ablation or task adaptation.

* For best results, review and balance label distributions in your dataset before preprocessing.

