from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from transformers import AutoTokenizer
import glob

tokenizer = AutoTokenizer.from_pretrained(
    "./models/Qwen2.5-Math-7B-Instruct-updated",
    use_fast = True
)

arrow_train = [Dataset.from_file(fp) for fp in sorted(glob.glob("<Train data path (arrow formatted)>"))]
raw_train_datasets = concatenate_datasets(arrow_train)

arrow_test = [Dataset.from_file(fp) for fp in sorted(glob.glob("<test data path (arrow formatted)>"))]
raw_test_datasets = concatenate_datasets(arrow_test)


column_names = list(raw_train_datasets.features)

def batch_preprocess(batch):
    input_convs = batch['inputs']
    label_convs = batch['labels']

    input_enc = tokenizer.apply_chat_template(
        input_convs,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        padding=False,
        max_length=tokenizer.model_max_length,
        return_dict=True
    )

    label_enc = tokenizer.apply_chat_template(
        label_convs,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        padding=False,
        max_length=tokenizer.model_max_length,
        return_dict=True
    )

    input_ids = input_enc["input_ids"]
    label_ids = label_enc["input_ids"]

    labels = [
        [-100 if i == j else j for i, j in zip(in_ids, lbl_ids)]
        for in_ids, lbl_ids in zip(input_ids, label_ids)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": input_enc["attention_mask"],
        "labels": labels
    }

train_dataset = raw_train_datasets.map(
    batch_preprocess, 
    batched=True, 
    remove_columns=column_names,
    num_proc=None,
    batch_size=256
    )

eval_dataset = raw_test_datasets.map(
    batch_preprocess, 
    batched=True, 
    remove_columns=column_names,
    num_proc=None,
    batch_size=256
    )

# train_dataset = raw_datasets["train"].map(
#     batch_preprocess, 
#     batched=True, 
#     remove_columns=column_names,
#     num_proc=None,
#     batch_size=256
#     )

# eval_dataset = raw_datasets["test"].map(
#     batch_preprocess, 
#     batched=True, 
#     remove_columns=column_names,
#     num_proc=None,
#     batch_size=256
#     )

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": eval_dataset
})

save_path = "./train_data/affordance_PRM_tokenized_qwen_25_7B"

# Save the DatasetDict to disk
dataset_dict.save_to_disk(save_path)

print(f"Combined dataset saved to: {save_path}")
