from typing import List, Dict


class DynamicCollator:
    """
    A collator class designed for dynamic padding of batches for use with
    large language model (LLM) inference. This class allows configurable
    padding behavior, including padding from the left or right side of sequences.

    Attributes:
        pad_token (int): The token used to pad sequences. Defaults to 1.
        pad_from_left (bool): Determines if padding is applied from the left side
            of the sequences. Defaults to False.
    """

    def __init__(
        self,
        pad_token: int = 1,
        pad_from_left: bool = False,
    ):
        """
        Initializes the DynamicCollator with padding configurations.

        Args:
            pad_token (int): The token used to pad sequences. Defaults to 1.
            pad_from_left (bool): If True, pads sequences from the left side.
                Defaults to False.
        """
        self.pad_token = pad_token
        self.pad_from_left = pad_from_left

    def custom_collate_fn(
        self, data: Dict[str, List[List[int]]]
    ) -> Dict[str, List[List[int]]]:
        """
        Custom collate function that pads sequences dynamically to the same length
        within a batch. Padding direction is controlled by 'pad_from_left'.

        Args:
            data (dict): A dictionary containing batch data with keys:
                - 'input_ids': List of token ID lists for each sample.
                - 'attention_mask': List of attention masks for each sample.

        Returns:
            dict: A dictionary with padded 'input_ids' and 'attention_mask', and
            placeholders for 'word_ids' and 'text'.
        """
        if not self.pad_from_left:
            max_length = len(data["input_ids"][-1])
            data["input_ids"] = [
                lst + [self.pad_token] * (max_length - len(lst))
                for lst in data["input_ids"]
            ]
            data["attention_mask"] = [
                lst + [0] * (max_length - len(lst))
                for lst in data["attention_mask"]
            ]
        else:
            max_length = len(data["input_ids"][-1])
            data["input_ids"] = [
                [self.pad_token] * (max_length - len(lst)) + lst
                for lst in data["input_ids"]
            ]
            data["attention_mask"] = [
                [0] * (max_length - len(lst)) + lst
                for lst in data["attention_mask"]
            ]

        return data

    def custom_collate_fn_no_dynamic(self, data):
        return {"splited": [element["splited"] for element in data]}
