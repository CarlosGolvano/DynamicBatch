"""
Example of how to use DynamicBatch.
"""

import torch
import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from DynamicBatch import (
    DatasetForDynamicBatch,
    DynamicCollator,
    DynamicBatchSampler,
)


tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
dataset = datasets.load_dataset("lhoestq/conll2003")["test"]


def tokenize(examples):
    # Very important to tokenize with padding False. DybamicCollator will
    # tokenize during inference.
    tokenized_examples = tokenizer(
        examples["tokens"],
        padding=False,
        is_split_into_words=True,
    )

    return tokenized_examples


# Tokenize the dataset
dataset = dataset.map(tokenize, batched=True)

# This dataset class implements a function to sort indices by length.
dataset_for_prediction = DatasetForDynamicBatch(dataset)

# Collator will pad inputs during inference and needs to know the pad_token
# and the padding side.
dynamic_collator = DynamicCollator(
    pad_token=tokenizer.pad_token_id, pad_from_left=True
)

# Max tokens per batch is the maximun of tokens that you want to be in GPU.
# Depending on the model, the max input size can be 512 (BERT), 1024 (BART)
# and much bigger for GPT, Llama, etc. models. The maximun amount of tokens
# will be equal to the bigger matrix of (batch_size x model max input size)
# that can fit you GPU.
batch_size = 32
max_tokens_per_batch = model.config.max_position_embeddings * batch_size
dynamic_batch_sampler = DynamicBatchSampler(
    indices=dataset_for_prediction.sort_indices(),
    input_ids=dataset_for_prediction["input_ids"],
    max_tokens_per_batch=max_tokens_per_batch,
)
dataloader = DataLoader(
    dataset_for_prediction,
    collate_fn=dynamic_collator.custom_collate_fn,
    batch_sampler=dynamic_batch_sampler,
    shuffle=False,
)

device = "cuda"
model = model.to(device)
model.eval()

for batch in dataloader:
    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor(batch["input_ids"], device=device),
            attention_mask=torch.tensor(
                batch["attention_mask"], device=device
            ),
        )
    logits = outputs[0]
    predictions = torch.argmax(logits, dim=1).tolist()
    for prediction in predictions:
        decoded_prediction = tokenizer.decode(prediction)
        print(decoded_prediction)
