import logging
import math
from typing import List, Dict, Union, Tuple, Any

import torch
import transformers as tr

from transformer_embedder import MODELS_WITH_STARTING_TOKEN, MODELS_WITH_DOUBLE_SEP


class Tokenizer:
    def __init__(self, model_name: str):
        # init huggingface tokenizer
        self.tokenizer = tr.AutoTokenizer.from_pretrained(model_name)
        # get config
        config = tr.AutoConfig.from_pretrained(model_name)
        # set the token type id
        self.token_type_id = self._get_token_type_id(config)

    def __call__(
        self,
        text: Union[List[List[str]], List[str]],
        text_pair: Union[List[List[str]], List[str]] = None,
        padding: bool = False,
        max_length: int = 0,
        return_tensor: bool = False,
        *args,
        **kwargs,
    ):
        if isinstance(text, str):
            raise ValueError(
                "`text` field is of type str. Pass a tokenized sentence to use this method"
            )
        # get model max length if not specified by user
        if max_length == 0:
            max_length = self.tokenizer.model_max_length

        # check if input is batched or a single sample
        is_batched = bool(text and isinstance(text[0], (list, tuple)))
        if not is_batched:
            output = self.build_tokens(text, text_pair, max_length)
        else:
            if not padding and return_tensor:
                logging.info(
                    f"""`padding` is False and return_tensor is True. Cannot make tensors from not padded sequences. 
                    `padding` is set automatically to True"""
                )
                padding = True
            output = self.build_tokens_batch(text, text_pair, max_length, padding)
        if return_tensor:
            output = self.to_tensor(output)
        return output

    def build_tokens_batch(
        self,
        text: List[List[str]],
        text_pair: List[List[str]] = None,
        max_len: int = math.inf,
        padding: bool = False,
    ) -> Union[
        Dict[str, torch.Tensor], List[Dict[str, Union[list, List[Tuple[int, int]], List[bool]]]]
    ]:
        batch = []
        if not text_pair:
            # In this way we can re-use the already defined methods, regardless the presence of the text pairs
            text_pair = [None for _ in text]
        for t, t_p in zip(text, text_pair):
            token_pair = self.build_tokens(t, t_p, max_len)
            batch.append(token_pair)
        # convert to dict
        if padding:
            batch = self.pad_batch(batch)
        return batch

    def build_tokens(
        self, text: List[str], text_pair: List[str] = None, max_len: int = math.inf
    ) -> Dict[str, Union[list, int]]:
        """
        Build transformer pair input
        :param text: sentence A
        :param text_pair: sentence B
        :param max_len: max_len of the sequence
        :return: a dictionary with A and B encoded
        """
        input_ids, token_type_ids, offsets = self._build_tokens(text, max_len=max_len)
        len_pair = len(text) + 2
        if text_pair:
            input_ids_b, token_type_ids_b, offsets_b = self._build_tokens(text_pair, True, max_len)
            # align offsets of sentence b
            offsets_b = [(o[0] + len(input_ids), o[1] + len(input_ids)) for o in offsets_b]
            offsets = offsets + offsets_b
            input_ids += input_ids_b
            token_type_ids += token_type_ids_b
            len_pair += len(text_pair) + 1

        attention_mask = [True] * len(input_ids)
        return {
            "input_ids": input_ids,
            "offsets": offsets,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "sentence_length": len_pair,
        }

    def _build_tokens(
        self, sentence: List[str], is_b: bool = False, max_len: int = math.inf
    ) -> Tuple[list, list, List[Tuple[int, int]]]:
        """
        Encode the sentence for BERT
        :param sentence: sentence to encode
        :param is_b: if it's the second sentence pair, skips first CLS token
                and set token_type_id to 1
        :return: encoded sentence
        """
        input_ids, token_type_ids, offsets = [], [], []
        if not is_b:
            token_type_id = 0
            # some models don't need starting special token
            if isinstance(self.tokenizer, MODELS_WITH_STARTING_TOKEN):
                input_ids += [self.tokenizer.cls_token_id]
                token_type_ids += [token_type_id]
                # first offset
                offsets.append((1, 1))
        else:
            token_type_id = self.token_type_id
            # check if the input needs an additional sep token
            # XLM-R for example wants an additional `</s>` between text pairs
            if isinstance(self.tokenizer, MODELS_WITH_DOUBLE_SEP):
                input_ids += [self.tokenizer.sep_token_id]
                token_type_ids += [token_type_id]
                offsets.append((len(input_ids), len(input_ids) + 1))
        for w in sentence:
            ids = self.tokenizer(w, add_special_tokens=False)["input_ids"]
            # if max_len exceeded, stop (leave space for closing token)
            if len(input_ids) + len(ids) >= max_len - 1:
                break
            # token offset before wordpiece, (start, end + 1)
            offsets.append((len(input_ids), len(input_ids) + len(ids) - 1))
            input_ids += ids
            token_type_ids += [token_type_id] * len(ids)
        # last offset
        offsets.append((len(input_ids), len(input_ids)))
        input_ids += [self.tokenizer.sep_token_id]
        token_type_ids += [token_type_id]
        return input_ids, token_type_ids, offsets

    def pad_batch(
        self, batch: List[Dict[str, Union[list, int]]]
    ) -> List[Dict[str, Union[list, List[bool], List[Tuple[int, int]]]]]:
        """
        Pad the batch to its maximum length
        :param batch: the batch to pad
        :return: the padded batch
        """
        # get maximum len inside a batch
        wp_max_batch_len = max(len(x["input_ids"]) for x in batch)
        word_max_batch_len = max(x["sentence_length"] for x in batch)
        for b in batch:
            input_ids_len = len(b["input_ids"])
            pad_len = wp_max_batch_len - input_ids_len
            word_pad_len = word_max_batch_len - b["sentence_length"]
            # for pad offset must be (0, 0)
            b["offsets"] += [(0, 0) for _ in range(word_pad_len)]
            b["input_ids"] += [self.tokenizer.pad_token_id] * pad_len
            b["attention_mask"] += [False] * pad_len
            b["token_type_ids"] += [1] * pad_len
        return batch

    @staticmethod
    def to_tensor(batch: Union[List[dict], dict]) -> Dict[str, torch.Tensor]:
        """
        Return a the batch in input as Pytorch tensors
        :param batch: batch in input
        :return: the batch as tensor
        """
        # single sentence case, generalize
        if isinstance(batch, dict):
            batch = [batch]
        # convert list to dict
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        batch = {
            "input_ids": torch.tensor(batch["input_ids"]),
            "offsets": torch.tensor(batch["offsets"]),
            "attention_mask": torch.tensor(batch["attention_mask"]),
            "token_type_ids": torch.tensor(batch["token_type_ids"]),
        }
        return batch

    @staticmethod
    def _get_token_type_id(config):
        if hasattr(config, "type_vocab_size"):
            return 1 if config.type_vocab_size == 2 else 0
        else:
            return 0
