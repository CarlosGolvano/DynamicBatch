from typing import Optional, List, Union
from datasets import Dataset
from torch.utils import data


class DatasetForDynamicBatch(data.Dataset):
    """
    A dataset wrapper designed for dynamic batching of data, particularly
    optimized for large language model (LLM) inference. This class allows
    flexible handling of datasets by providing methods to retrieve single
    or multiple items and to sort indices based on input sequence lengths,
    which is useful for creating efficient batches.

    Attributes:
        data (Dataset): The dataset to wrap for dynamic batching.
        cache_file_name (Optional[str]): Optional file name for caching the
            dataset if required.
    """

    def __init__(
        self,
        data: Dataset,
        cache_file_name: Optional[str] = None,
    ):
        """
        Initializes the DatasetForDynamicBatch with the provided dataset
        and an optional cache file name.

        Args:
            data (Dataset): The dataset object to use, typically containing
                data for LLM inference.
            cache_file_name (Optional[str]): Optional cache file name to
                use for saving processed data.

        Raises:
            TypeError: If 'data' is not an instance of 'Dataset'.
        """
        super().__init__()
        if not isinstance(data, Dataset):
            raise TypeError(
                f"data type must be {Dataset}, not " f"{data.__class__}"
            )
        self.data = data
        self.cache_file_name = cache_file_name

    def sort_indices(self, reverse=False) -> Optional[List[int]]:
        """
        Sorts and returns indices of the dataset based on the length of
        the 'input_ids' feature, useful for dynamic batching where sequences
        need to be grouped by length.

        Args:
            reverse (bool): If True, sorts indices in descending order of
                sequence lengths. Defaults to False.

        Returns:
            list or None: A list of sorted indices if 'input_ids' is a
            feature in the dataset, otherwise None.
        """
        # TODO: remove polymorphic return type
        sorted_indices = None
        if "input_ids" in self.data.features.keys():
            data_lenghts = [len(x) for x in self.data["input_ids"]]
            sorted_indices = sorted(
                range(len(data_lenghts)),
                key=lambda i: data_lenghts[i],
                reverse=reverse,
            )

        return sorted_indices

    def __getitem__(self, index) -> dict:
        return self.data[index]

    def __getitems__(self, indices) -> List[dict]:
        return self.data[indices]

    def __repr__(self) -> str:
        msg = f"DatasetEmbd(\n  {self.data.__repr__()})"
        return msg

    def __len__(self) -> int:
        return len(self.data)
