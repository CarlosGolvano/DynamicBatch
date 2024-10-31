# Dynamic Batch Inference Library for LLMs

This repository provides a Python library designed for dynamic batching of large language model (LLM) inference. This library includes classes for dynamically padding and batching sequences of variable lengths, optimized to work efficiently within specified token limits. It enables faster and more memory-efficient inference through smart batching strategies that are easy to integrate with PyTorch data pipelines.

## Features

- **Dynamic Padding**: Dynamically pads sequences within batches, with configurable padding direction.
- **Dynamic Batch Sampler**: Creates batches based on a configurable token limit to improve batch utilization.
- **Custom Collators**: Provides custom collate functions for PyTorch's DataLoader, supporting variable-length sequences.

## Requirements

- Python 3.7 or later
- PyTorch
- Datasets library by HuggingFace🤗

You can install the necessary libraries with the following command:

```bash
pip install torch datasets transformers
```

## Installation

Clone this repository and install the library:

```bash
git clone https://github.com/CarlosGolvano/DynamicBatch.git
cd DynamicBatch
pip install -e .
```

## Usage

Below is an example usage of the `DatasetForDynamicBatch`, `DynamicBatchSampler`, and `DynamicCollator` classes to create dynamically padded and batched data for LLM inference. There is also an example script in `examples` folder.

### Step 1: Initialize the Dataset

Wrap your tokenized dataset with the `DatasetForDynamicBatch` class.

```python
from datasets import load_dataset
from DynamicBatch import DatasetForDynamicBatch

dataset = load_dataset("your_dataset")
tokenized_dataset = tokenizer(dataset)
dynamic_dataset = DatasetForDynamicBatch(tokenized_dataset['test'])
```

### Step 2: Set Up the Sampler

Use the `DynamicBatchSampler` to create batches that respect a maximum token limit. This sampler batches sequences by considering both sequence length and token count, maximizing efficiency. The `DatasetForDynamicBatch` class provides options for sorting by sequence length for improved batching efficiency.

```python
from DynamicBatch import DynamicBatchSampler

batch_size = 32
max_tokens_per_batch = model.config.max_position_embeddings * batch_size
dynamic_batch_sampler = DynamicBatchSampler(
    indices=dataset_for_prediction.sort_indices(reverse=False),
    input_ids=dataset_for_prediction["input_ids"],
    max_tokens_per_batch=max_tokens_per_batch,
)
```

### Step 3: Set Up the DataLoader with Dynamic Collator

The `DynamicCollator` handles padding within batches, with options to pad from the left or right. This allows seamless batching of variable-length sequences.

```python
from torch.utils.data import DataLoader
from DynamicBatch import DynamicCollator

dynamic_collator = DynamicCollator(
    pad_token=tokenizer.pad_token_id, pad_from_left=True
)
dataloader = DataLoader(
    dataset_for_prediction,
    collate_fn=dynamic_collator.custom_collate_fn,
    batch_sampler=dynamic_batch_sampler,
    shuffle=False,
)

# Iterate through batches
for batch in dataloader:
    # Perform model inference
    outputs = model(batch['input_ids'])
```

## Contributing

Contributions are welcome! If you'd like to improve or add features, feel free to fork the repository, create a new branch, and open a pull request.

    Fork the repository
    Create your feature branch (git checkout -b feature/YourFeature)
    Commit your changes (git commit -m 'Add YourFeature')
    Push to the branch (git push origin feature/YourFeature)
    Open a pull request

### License

This project is licensed under the MIT License - see the LICENSE file for details.

