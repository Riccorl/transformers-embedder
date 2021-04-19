import math
from collections import UserDict
from functools import partial
from typing import List, Dict, Union, Tuple, Any

import spacy
import torch
import transformers as tr
from spacy.cli.download import download as spacy_download

from transformer_embedder import MODELS_WITH_STARTING_TOKEN, MODELS_WITH_DOUBLE_SEP
from transformer_embedder import utils

logger = utils.get_logger(__name__)
utils.get_logger("transformers")


class Tokenizer:
    """Tokenizer class."""

    def __init__(
        self,
        model: Union[str, tr.PreTrainedTokenizer],
        language: str = "xx_sent_ud_sm",
    ):
        if isinstance(model, str):
            # init huggingface tokenizer
            self.huggingface_tokenizer = tr.AutoTokenizer.from_pretrained(model)
            # get config
            self.config = tr.AutoConfig.from_pretrained(model)
        else:
            self.huggingface_tokenizer = model
            self.config = tr.AutoConfig.from_pretrained(self.huggingface_tokenizer.name_or_path)
        # spacy tokenizer, lazy load. None at first
        self.spacy_tokenizer = None
        # default multilingual model
        self.language = language
        # padding stuff
        # default, batch length is model max length
        self.subtoken_max_batch_len = self.huggingface_tokenizer.model_max_length
        self.word_max_batch_len = self.huggingface_tokenizer.model_max_length
        # padding ops
        self.padding_ops = {
            "input_ids": partial(
                self.pad_sequence,
                value=self.huggingface_tokenizer.pad_token_id,
                length="subtoken",
            ),
            "offsets": partial(self.pad_sequence, value=(-1, -1), length="word"),
            "attention_mask": partial(self.pad_sequence, value=0, length="subtoken"),
            "word_mask": partial(self.pad_sequence, value=0, length="word"),
            "token_type_ids": partial(
                self.pad_sequence,
                value=self.token_type_id,
                length="subtoken",
            ),
        }
        # keys that will be converted in tensors
        self.to_tensor_inputs = {
            "input_ids",
            "offsets",
            "attention_mask",
            "word_mask",
            "token_type_ids",
        }

    def __len__(self):
        """Size of the full vocabulary with the added tokens."""
        return len(self.huggingface_tokenizer)

    def __call__(
        self,
        text: Union[List[List[str]], List[str], str],
        text_pair: Union[List[List[str]], List[str], str] = None,
        padding: bool = False,
        max_length: int = 0,
        is_split_into_words: bool = False,
        use_spacy: bool = False,
        return_tensors: bool = False,
        *args,
        **kwargs,
    ):
        """
        Prepare the text in input for the :obj:`TransformerEmbedder` module.

        Args:
            text: Text or batch of text to be encoded.
            text_pair: Text or batch of text to be encoded.
            padding: If True, applies padding to the batch based on the maximum length of the batch.
            max_length: If specified, truncates the input sequence to that value. Otherwise,
            uses the model max length.
            is_split_into_words: If True and the input is a string, the input is split on spaces.
            use_spacy: If True, use :obj:`self.spacy_tokenizer`
            return_tensors: If True, the outputs is converted to :obj:`torch.Tensor`
            *args:
            **kwargs:

        Returns:
            The input for the model as dictionary with the following keys:
            "input_ids",
            "offsets",
            "attention_mask",
            "token_type_ids",
            "sentence_length"

        """
        # type checking before everything
        self._type_checking(text, text_pair)
        if isinstance(text, str) and not use_spacy:
            logger.warning(
                "`text` field is of type `str`, splitting by spaces. "
                "Set `use_spacy` to tokenize using spacy model."
            )

        # check if input is batched or a single sample
        is_batched = bool(
            (not is_split_into_words and isinstance(text, (list, tuple)))
            or (
                is_split_into_words
                and isinstance(text, (list, tuple))
                and text
                and isinstance(text[0], (list, tuple))
            )
        )

        # if text is str or a list of str and they are not split, then text needs to be tokenized
        if isinstance(text, str) or (not is_split_into_words and isinstance(text[0], str)):
            if not is_batched:
                text = self.pretokenize(text, use_spacy=use_spacy)
                text_pair = self.pretokenize(text_pair, use_spacy=use_spacy) if text_pair else None
            else:
                text = [self.pretokenize(t, use_spacy=use_spacy) for t in text]
                text_pair = (
                    [self.pretokenize(t, use_spacy=use_spacy) for t in text_pair]
                    if text_pair
                    else None
                )

        # get model max length if not specified by user
        if max_length == 0:
            max_length = self.huggingface_tokenizer.model_max_length

        if not is_batched:
            output = self.build_tokens(text, text_pair, max_length)
        else:
            if not padding and return_tensors:
                logger.info(
                    "`padding` is False and return_tensor is True. Cannot make tensors from "
                    "not padded sequences. `padding` forced automatically to True"
                )
                padding = True
            output = self.build_tokens_batch(text, text_pair, max_length)

        # clean the output
        output = self._clean_output(output)

        # pad batch
        if padding:
            output = self.pad_batch(output)

        # convert to tensor
        if return_tensors:
            output = self.to_tensor(output)

        output = ModelInputs(output)
        return output

    def build_tokens_batch(
        self,
        text: List[List[str]],
        text_pair: List[List[str]] = None,
        max_len: int = math.inf,
    ) -> Union[
        Dict[str, torch.Tensor],
        List[Dict[str, Union[list, List[Tuple[int, int]], List[bool]]]],
    ]:
        """
        Builds the batched input.

        Args:
            text (List[List[str]]): Text or batch of text to be encoded.
            text_pair (List[List[str]] or None): Text pair or batch of text to be encoded.
            max_len (int): If specified, truncates the input sequence to that value. Otherwise,
            uses the model max length.

        Returns:
            The encoded batch

        """
        batch = []
        if not text_pair:
            # In this way we can re-use the already defined methods,
            # regardless the presence of the text pairs
            text_pair = [None for _ in text]
        for t, t_p in zip(text, text_pair):
            token_pair = self.build_tokens(t, t_p, max_len)
            batch.append(token_pair)
        return batch

    def build_tokens(
        self, text: List[str], text_pair: List[str] = None, max_len: int = math.inf
    ) -> Dict[str, Union[list, int]]:
        """
        Build transformer pair input.

        Args:
            text (): sentence A
            text_pair (): sentence B
            max_len (): max length of the sequence

        Returns:
            a dictionary with A and B encoded
        """
        input_ids, token_type_ids, offsets = self._build_tokens(text, max_len=max_len)
        len_pair = len(text) + (
            2 if isinstance(self.huggingface_tokenizer, MODELS_WITH_STARTING_TOKEN) else 1
        )
        if text_pair:
            input_ids_b, token_type_ids_b, offsets_b = self._build_tokens(text_pair, True, max_len)
            # align offsets of sentence b
            offsets_b = [(o[0] + len(input_ids), o[1] + len(input_ids)) for o in offsets_b]
            offsets = offsets + offsets_b
            input_ids += input_ids_b
            token_type_ids += token_type_ids_b
            len_pair += len(text_pair) + (
                2 if isinstance(self.huggingface_tokenizer, MODELS_WITH_DOUBLE_SEP) else 1
            )

        word_mask = [1] * len_pair  # for original tokens
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "offsets": offsets,
            "attention_mask": attention_mask,
            "word_mask": word_mask,
            "token_type_ids": token_type_ids,
            "sentence_length": len_pair,
        }

    def _build_tokens(
        self, sentence: List[str], is_b: bool = False, max_len: int = math.inf
    ) -> Tuple[list, list, List[Tuple[int, int]]]:
        """
        Encode the sentence for transformer model.

        Args:
            sentence (List[str]): sentence to encode
            is_b (bool): if it's the second sentence pair, skips first CLS token and set token_type_id to 1
            max_len (int):

        Returns:
            Tuple: The encoded sentence.
        """
        input_ids, token_type_ids, offsets = [], [], []
        if not is_b:
            token_type_id = 0
            # some models don't need starting special token
            if isinstance(self.huggingface_tokenizer, MODELS_WITH_STARTING_TOKEN):
                input_ids += [self.huggingface_tokenizer.cls_token_id]
                token_type_ids += [token_type_id]
                # first offset
                offsets.append((0, 0))
        else:
            token_type_id = self.token_type_id
            # check if the input needs an additional sep token
            # XLM-R for example wants an additional `</s>` between text pairs
            if isinstance(self.huggingface_tokenizer, MODELS_WITH_DOUBLE_SEP):
                input_ids += [self.huggingface_tokenizer.sep_token_id]
                token_type_ids += [token_type_id]
                offsets.append((len(input_ids) - 1, len(input_ids) - 1))
        for w in sentence:
            ids = self.huggingface_tokenizer(w, add_special_tokens=False)["input_ids"]
            # if max_len exceeded, stop (leave space for closing token)
            if len(input_ids) + len(ids) >= max_len - 1:
                break
            # token offset before wordpiece, (start, end + 1)
            offsets.append((len(input_ids), len(input_ids) + len(ids) - 1))
            input_ids += ids
            token_type_ids += [token_type_id] * len(ids)
        # last offset
        offsets.append((len(input_ids), len(input_ids)))
        input_ids += [self.huggingface_tokenizer.sep_token_id]
        token_type_ids += [token_type_id]
        return input_ids, token_type_ids, offsets

    def pad_batch(self, batch: Dict[str, list], max_length: int = None) -> Dict[str, list]:
        """
        Pad the batch to its maximum length.

        Args:
            batch (Dict[str, list]): the batch to pad
            max_length (int): override maximum length of the batch

        Returns:
            Dict[str, list]: The padded batch
        """
        if max_length:
            self.subtoken_max_batch_len = max_length
            self.word_max_batch_len = max_length
        else:
            # get maximum len inside a batch
            self.subtoken_max_batch_len = max(len(x) for x in batch["input_ids"])
            self.word_max_batch_len = max(x for x in batch["sentence_length"])
        for key in batch.keys():
            if key in self.padding_ops.keys():
                batch[key] = [self.padding_ops[key](b) for b in batch[key]]
        return batch

    def pad_sequence(
        self,
        sequence: Union[List, torch.Tensor],
        value: Any,
        length: Union[int, str] = "subtoken",
        pad_to_left: bool = False,
    ) -> Union[List, torch.Tensor]:
        """
        Pad the input to the specified length with the given value.

        Args:
            sequence (Union[List, torch.Tensor]):  element to pad, it can be either a :obj:`List`
            or a :obj:`torch.Tensor`
            value (Any): value to use as padding
            length (Union[int, str]): length after pad
            pad_to_left (bool): if True, pads to the left, right otherwise

        Returns:
            Union[List, torch.Tensor]: the padded sequence

        """
        if length == "subtoken":
            length = self.subtoken_max_batch_len
        elif length == "word":
            length = self.word_max_batch_len
        else:
            if not isinstance(length, int):
                raise ValueError(
                    f"`length` must be an `int`, `subtoken` or `word`. Current value is `{length}`"
                )
        padding = [value] * abs(length - len(sequence))
        if isinstance(sequence, torch.Tensor):
            if len(sequence.shape) > 1:
                raise ValueError(
                    f"Sequence tensor must be 1D. Current shape is `{len(sequence.shape)}`"
                )
            padding = torch.as_tensor(padding)
        if pad_to_left:
            if isinstance(sequence, torch.Tensor):
                return torch.cat((padding, sequence), -1)
            return padding + sequence
        if isinstance(sequence, torch.Tensor):
            return torch.cat((sequence, padding), -1)
        return sequence + padding

    def pretokenize(self, text: str, use_spacy: bool = False) -> List[str]:
        """
        Pre-tokenize the text in input, splitting on spaces or using SpaCy tokenizer if `use_spacy` is True.
        Args:
            text (str): text to tokenize
            use_spacy (bool): if True, uses a SpaCy tokenizer

        Returns:
            List[str]: the tokenized text

        """
        if use_spacy:
            if not self.spacy_tokenizer:
                self._load_spacy()
            text = self.spacy_tokenizer(text)
            return [t.text for t in text]
        return text.split(" ")

    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, tr.AddedToken]]) -> int:
        """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder.
        If special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last
        index of the current vocabulary).

        Args:
            special_tokens_dict (`dict`): the dictionary containing special tokens. Keys should be in
            the list of predefined special attributes: [``bos_token``, ``eos_token``,
            ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
            ``additional_special_tokens``].

        Returns:
            int: Number of tokens added to the vocabulary.

        """
        return self.huggingface_tokenizer.add_special_tokens(special_tokens_dict)

    def add_padding_ops(self, key: str, value: Any, length: Union[int, str]):
        """
        Add padding logic to custom fields.
        If the field is not in `self.to_tensor_inputs`, this method will add the key to it.

        Args:
            key (str): name of the field in the tokenzer input
            value (Any): value to use for padding
            length (int or str): length to pad. It can be an `int`, or two string value
            `subtoken`: the element is padded to the batch max length relative to the subtokens length
            `word`: the element is padded to the batch max length relative to the original word length

        Returns:

        """
        if key not in self.to_tensor_inputs:
            self.to_tensor_inputs.add(key)
        self.padding_ops[key] = partial(self.pad_sequence, value=value, length=length)

    def add_to_tensor_inputs(self, names: Union[str, set]):
        """
        Add these keys to the ones that will be converted in Tensors.

        Args:
            names (`str` or `set`): name of the field (or fields) to convert to tensors

        Returns:

        """
        if isinstance(names, str):
            names = {names}
        self.to_tensor_inputs |= names

    def to_tensor(self, batch: Union[List[dict], dict]) -> Dict[str, torch.Tensor]:
        """
        Return a the batch in input as Pytorch tensors.
        The fields that are converted in tensors are in `self.to_tensor_inputs`. By default, only the
        standard model inputs are converted. Use 'Tokenizer.add_to_tensor_inputs` to add custom fields.

        Args:
            batch (List[dict] or dict): batch in input

        Returns:
            Dict: the batch as tensor

        """
        # convert to tensor
        batch = {
            k: torch.as_tensor(v) if k in self.to_tensor_inputs else v for k, v in batch.items()
        }
        return batch

    def _load_spacy(self) -> spacy.tokenizer.Tokenizer:
        """
        Download and load spacy model.

        Returns:
            spacy.tokenizer.Tokenizer: The spacy tokenizer loaded.

        """
        try:
            spacy_tagger = spacy.load(self.language, exclude=["ner", "parser"])
        except OSError:
            logger.info(f"Spacy model '{self.language}' not found. Downloading and installing.")
            spacy_download(self.language)
            spacy_tagger = spacy.load(self.language, exclude=["ner", "parser"])
        self.spacy_tokenizer = spacy_tagger.tokenizer
        return self.spacy_tokenizer

    @staticmethod
    def _clean_output(output: Union[List, Dict]) -> Dict:
        """
        Clean before output.

        Args:
            output (Union[List, Dict]): the output to clean

        Returns:
            Dict: the cleaned output

        """
        # single sentence case, generalize
        if isinstance(output, dict):
            output = [output]
        # convert list to dict
        output = {k: [d[k] for d in output] for k in output[0]}
        return output

    @staticmethod
    def _get_token_type_id(config) -> int:
        """
        Get token type id. Useful when dealing with models that don't accept 1 as type id.
        Args:
            config (): transformer config

        Returns:
            int: correct token tyoe id for that model

        """
        if hasattr(config, "type_vocab_size"):
            return 1 if config.type_vocab_size == 2 else 0
        return 0

    @staticmethod
    def _type_checking(text: Any, text_pair: Any):
        """
        Checks type of the inputs.

        Args:
            text ():
            text_pair ():

        Returns:

        """

        def is_type_correct(text_to_check: Any) -> bool:
            """
            Check if input type is correct, returning a boolean.

            Args:
                text_to_check (): text to check

            Returns:
                (bool): True if the type is correct

            """
            return (
                text_to_check is None
                or isinstance(text_to_check, str)
                or (
                    isinstance(text_to_check, (list, tuple))
                    and (
                        len(text_to_check) == 0
                        or (
                            isinstance(text_to_check[0], str)
                            or (
                                isinstance(text_to_check[0], (list, tuple))
                                and (
                                    len(text_to_check[0]) == 0
                                    or isinstance(text_to_check[0][0], str)
                                )
                            )
                        )
                    )
                )
            )

        if not is_type_correct(text):
            raise AssertionError(
                "text input must of type `str` (single example), `List[str]` (batch or single "
                "pretokenized example) or `List[List[str]]` (batch of pretokenized examples)."
            )

        if not is_type_correct(text_pair):
            raise AssertionError(
                "text_pair input must be `str` (single example), `List[str]` (batch or single "
                "pretokenized example) or `List[List[str]]` (batch of pretokenized examples)."
            )

    @property
    def num_special_tokens(self) -> int:
        """
        Return the number of special tokens the model needs.
        It assume the input contains both sentences (`text` and `text_pair`).

        Returns:
            int: the number of special tokens

        """
        if isinstance(self.huggingface_tokenizer, MODELS_WITH_DOUBLE_SEP) and isinstance(
            self.huggingface_tokenizer, MODELS_WITH_STARTING_TOKEN
        ):
            return 4
        if isinstance(
            self.huggingface_tokenizer,
            (MODELS_WITH_DOUBLE_SEP, MODELS_WITH_STARTING_TOKEN),
        ):
            return 3
        return 2

    @property
    def has_double_sep(self):
        """True if tokenizer uses two SEP tokens."""
        return isinstance(self.huggingface_tokenizer, MODELS_WITH_DOUBLE_SEP)

    @property
    def has_starting_token(self):
        """True if tokenizer uses a starting token."""
        return isinstance(self.huggingface_tokenizer, MODELS_WITH_STARTING_TOKEN)

    @property
    def token_type_id(self):
        """Padding token."""
        return self._get_token_type_id(self.config)

    @property
    def pad_token(self):
        """Padding token."""
        return self.huggingface_tokenizer.pad_token

    @property
    def pad_token_id(self):
        """Padding token id."""
        return self.huggingface_tokenizer.pad_token_id

    @property
    def unk_token(self):
        """Unknown token."""
        return self.huggingface_tokenizer.unk_token

    @property
    def unk_token_id(self):
        """Unknown token id."""
        return self.huggingface_tokenizer.unk_token_id

    @property
    def cls_token(self):
        """
        Classification token.
        To extract a summary of an input sequence leveraging self-attention along the
        full depth of the model.
        """
        return self.huggingface_tokenizer.cls_token

    @property
    def cls_token_id(self):
        """
        Classification token id.
        To extract a summary of an input sequence leveraging self-attention along the
        full depth of the model.
        """
        return self.huggingface_tokenizer.cls_token_id

    @property
    def sep_token(self):
        """Separation token, to separate context and query in an input sequence."""
        return self.huggingface_tokenizer.sep_token

    @property
    def sep_token_id(self):
        """Separation token id, to separate context and query in an input sequence."""
        return self.huggingface_tokenizer.sep_token_id

    @property
    def bos_token(self):
        """Beginning of sentence token."""
        return self.huggingface_tokenizer.bos_token

    @property
    def bos_token_id(self):
        """Beginning of sentence token id."""
        return self.huggingface_tokenizer.bos_token_id

    @property
    def eos_token(self):
        """End of sentence token."""
        return self.huggingface_tokenizer.eos_token

    @property
    def eos_token_id(self):
        """End of sentence token id."""
        return self.huggingface_tokenizer.eos_token_id


class ModelInputs(UserDict):
    """Model input dictionary wrapper."""

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getitem__(self, item: str) -> Any:
        return self.data[item]

    def __getstate__(self):
        return {"data": self.data}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

    def keys(self):
        """A set-like object providing a view on D's keys."""
        return self.data.keys()

    def values(self):
        """An object providing a view on D's values."""
        return self.data.values()

    def items(self):
        """A set-like object providing a view on D's items."""
        return self.data.items()

    def to(self, device: Union[str, "torch.device"]) -> "ModelInputs":
        """
        Send all tensors values to device.

        Args:
            device (:obj:`str` or :obj:`torch.device`): The device to put the tensors on.

        Returns:
            :class:`tokenizers.ModelInputs`: The same instance of
        :class:`~tokenizers.ModelInputs` after modification.

        """
        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, (str, torch.device, int)):
            self.data = {
                k: v.to(device=device) if isinstance(v, torch.Tensor) else v
                for k, v in self.data.items()
            }
        else:
            logger.warning(
                f"Attempting to cast to another type, {str(device)}. This is not supported."
            )
        return self
