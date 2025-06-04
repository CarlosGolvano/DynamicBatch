from torch.utils import data
from typing import List, Optional, Iterator


class DynamicBatchSampler(data.Sampler):
    """
    A dynamic batch sampler that generates batches of indices based on
    a maximum token limit per batch. This is optimized for batching sequences
    of variable lengths efficiently within a specified token budget.

    Attributes:
        indices (List[int]): List of dataset indices.
        input_ids (List[List[int]]): List of token ID sequences for each sample.
        max_tokens_per_batch (Optional[int]): The maximum number of tokens allowed
            in each batch. Defaults to 514 * 256.
    """

    def __init__(
        self,
        indices: List[int],
        input_ids: List[List[int]],
        max_tokens_per_batch: Optional[int] = 514 * 256,
    ):
        """
        Initializes the DynamicBatchSampler with dataset indices, token ID
        sequences, and an optional token budget per batch.

        Args:
            indices (List[int]): List of dataset indices to sample from.
            input_ids (List[List[int]]): List of token ID sequences.
            max_tokens_per_batch (Optional[int]): The maximum token count allowed
                per batch. Defaults to 514 * 256.
        """
        self.indices = indices
        self.input_ids = input_ids
        self.input_lengths = [len(ids) for ids in input_ids]
        self.max_tokens_per_batch = max_tokens_per_batch

    def __iter__(self) -> Iterator[List[int]]:
        """
        Generates batches of indices based on the maximum token limit, grouping
        indices such that the total tokens per batch does not exceed the limit.

        Yields:
            list of int: A batch of indices whose combined token count is within
            the specified token limit.
        """
        j = 0
        max_input_len = 0
        indices = self.indices
        lengths = self.input_lengths
        for i, idx in enumerate(indices):
            length = lengths[idx]
            if length > max_input_len:
                max_input_len = length

            if (i + 1 - j) * max_input_len > self.max_tokens_per_batch:
                yield indices[j:i]
                j = i
                max_input_len = length

            if i == len(indices) - 1:
                yield indices[j:]

    def __len__(
        self,
    ) -> int:
        return len(self.indices)
