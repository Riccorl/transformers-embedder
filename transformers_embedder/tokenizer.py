from __future__ import annotations

import math
from collections import UserDict
from functools import partial
from typing import List, Dict, Union, Tuple, Any

import transformers as tr

from transformers_embedder import MODELS_WITH_STARTING_TOKEN, MODELS_WITH_DOUBLE_SEP
from transformers_embedder import utils
from transformers_embedder.utils import is_torch_available, is_spacy_available

if is_torch_available():
    import torch

if is_spacy_available():
    import spacy
    from spacy.cli.download import download as spacy_download

logger = utils.get_logger(__name__)
utils.get_logger("transformers")


class Tokenizer:
    """
    A wrapper class for HuggingFace Tokenizer.

    Args:
        model (:obj:`str`, :obj:`transformers.PreTrainedTokenizer`):
            Language model name (or a transformer :obj:`PreTrainedTokenizer`.
    """

    def __init__(self, model: Union[str, tr.PreTrainedTokenizer], language: str = "xx_sent_ud_sm"):
        if isinstance(model, str):
            # init HuggingFace tokenizer
            self.huggingface_tokenizer = tr.AutoTokenizer.from_pretrained(model)
            # get config
            self.config = tr.AutoConfig.from_pretrained(model)
        else:
            self.huggingface_tokenizer = model
            self.config = tr.AutoConfig.from_pretrained(self.huggingface_tokenizer.name_or_path)
        # simple tokenizer used if the input is `str`
        # lazy load, None at first
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
            # value is None because: (read `pad_sequence` doc)
            "offsets": partial(self.pad_sequence, value=None, length="subtoken"),
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
        text: Union[str, List[str], List[List[str]]],
        text_pair: Union[str, List[str], List[List[str]]] = None,
        padding: bool = True,
        max_length: int = 0,
        return_tensors: bool = True,
        is_split_into_words: bool = False,
        use_spacy: bool = False,
        *args,
        **kwargs,
    ) -> ModelInputs:
        """
        Prepare the text in input for models that uses HuggingFace as embeddings.

        Args:
            text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`, :obj:`List[List[Word]]`, :obj:`List[Word]`):
                Text or batch of text to be encoded.
            text_pair (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`, :obj:`List[List[Word]]`, :obj:`List[Word]`):
                Text or batch of text to be encoded.
            padding (:obj:`bool`, optional, defaults to :obj:`True`):
                If :obj:`True`, applies padding to the batch based on the maximum length of the batch.
            max_length (:obj:`int`, optional, defaults to :obj:`0`):
                If specified, truncates the input sequence to that value. Otherwise,
                uses the model max length.
            return_tensors (:obj:`bool`, optional, defaults to :obj:`True`):
                If :obj:`True`, the outputs is converted to :obj:`torch.Tensor`
            is_split_into_words (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True` and the input is a string, the input is split on spaces.
            use_spacy (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True` the input is split using SpaCy, otherwise it is split on spaces.

        Returns:
            :obj:`ModelInputs`: The inputs to the transformer model.

        """
        # type checking before everything
        # self._type_checking(text, text_pair)

        # check if input is batched or a single sample
        is_batched = bool(isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple)))

        # if text is str or a list of str and they are not split, then text needs to be tokenized
        if isinstance(text, str) or (not is_split_into_words and isinstance(text[0], str)):
            if not is_batched:
                text = self.pretokenize(text, use_spacy=use_spacy)
                text_pair = self.pretokenize(text_pair, use_spacy=use_spacy) if text_pair else None
            else:
                text = [self.pretokenize(t, use_spacy=use_spacy) for t in text]
                text_pair = (
                    [self.pretokenize(t, use_spacy=use_spacy) for t in text_pair] if text_pair else None
                )

        # get model max length if not specified by user
        if max_length == 0:
            max_length = self.huggingface_tokenizer.model_max_length

        if not is_batched:
            output = self.build_tokens(text, text_pair, max_length)
        else:
            if not padding and return_tensors:
                logger.info(
                    "`padding` is `False` and return_tensor is `True`. Cannot make tensors from "
                    "not padded sequences. `padding` forced automatically to `True`"
                )
                padding = True
            output = self.build_tokens_batch(text, text_pair, max_length)

        # clean the output
        output = self._clean_output(output)
        if padding:
            output = self.pad_batch(output)
        if return_tensors:
            output = self.to_tensor(output)
        output = ModelInputs(output)
        return output

    def build_tokens_batch(
        self,
        text: List[List[str]],
        text_pair: List[List[str]] = None,
        max_length: int = math.inf,
    ) -> List[Dict[str, Union[list, int]]]:
        """
        Builds the batched input.

        Args:
            text (:obj:`List[List[Word]]`, :obj:`List[Word]`):
                Text or batch of text to be encoded.
            text_pair (:obj:`List[List[Word]]`, :obj:`List[Word]`):
                Text pair or batch of text to be encoded.
           max_length (:obj:`int`, optional, defaults to :obj:`0`):
                If specified, truncates the input sequence to that value. Otherwise,
                uses the model max length.

        Returns:
            :obj:`List[Dict[str, Union[list, int]]]`: The encoded batch

        """
        batch = []
        if not text_pair:
            # In this way we can re-use the already defined methods,
            # regardless the presence of the text pairs
            text_pair = [None for _ in text]
        for t, t_p in zip(text, text_pair):
            token_pair = self.build_tokens(t, t_p, max_length)
            batch.append(token_pair)
        return batch

    def build_tokens(
        self, text: List[str], text_pair: List[str] = None, max_length: int = math.inf
    ) -> Dict[str, Union[list, int]]:
        """
        Build transformer pair input.

        Args:
            text (:obj:`List[Word]`):
                Text to be encoded.
            text_pair (:obj:`List[Word]`):
                Text pair to be encoded.
           max_length (:obj:`int`, optional, defaults to :obj:`0`):
                If specified, truncates the input sequence to that value. Otherwise,
                uses the model max length.

        Returns:
            :obj:`Dict[str, Union[list, int]]`: A dictionary with :obj:`text` and :obj:`text_pair` encoded.
        """
        words, input_ids, token_type_ids, offsets = self._build_tokens(text, max_length=max_length)
        if text_pair:
            words_b, input_ids_b, token_type_ids_b, offsets_b = self._build_tokens(
                text_pair, True, max_length
            )
            # align offsets of sentence b
            offsets_b = [o + len(words) for o in offsets_b]
            offsets = offsets + offsets_b
            input_ids += input_ids_b
            token_type_ids += token_type_ids_b
            words += words_b

        word_mask = [1] * len(words)  # for original tokens
        attention_mask = [1] * len(input_ids)

        return {
            "words": words,
            "input_ids": input_ids,
            "offsets": offsets,
            "attention_mask": attention_mask,
            "word_mask": word_mask,
            "token_type_ids": token_type_ids,
            "sentence_lengths": len(words),
        }

    def _build_tokens(
        self, text: List[str], is_b: bool = False, max_length: int = math.inf
    ) -> Tuple[list, list, list, list]:
        """
        Encode the sentence for transformer model.

        Args:
            text (:obj:`List[Word]`):
                Text to encode.
            is_b (:obj:`bool`, optional, defaults to :obj:`False`):
                if :obj:`True`, skips first :obj:`CLS` token and set token_type_id to :obj:`1`
            max_length (:obj:`int`, optional, defaults to :obj:`0`):
                If specified, truncates the input sequence to that value. Otherwise,
                uses the model max length.

        Returns:
            :obj:`Tuple`:
                The encoded sentence, with :obj:`input_ids`, :obj:`token_type_ids` and :obj:`offsets`.
        """
        # words
        words = []
        # HuggingFace model inputs
        input_ids = []
        token_type_ids = []
        offsets = []
        offset_starting_index = 0
        if not is_b:
            token_type_id = 0
            # some models don't need starting special token
            if isinstance(self.huggingface_tokenizer, MODELS_WITH_STARTING_TOKEN):
                words += [self.huggingface_tokenizer.cls_token]
                input_ids += [self.huggingface_tokenizer.cls_token_id]
                token_type_ids += [token_type_id]
                # first offset
                offsets.append(0)
                offset_starting_index = 1
        else:
            token_type_id = self.token_type_id
            # check if the input needs an additional sep token
            # XLM-R for example wants an additional `</s>` between text pairs
            if isinstance(self.huggingface_tokenizer, MODELS_WITH_DOUBLE_SEP):
                words += [self.huggingface_tokenizer.sep_token]
                input_ids += [self.huggingface_tokenizer.sep_token_id]
                token_type_ids += [token_type_id]
                offsets.append(0)
                offset_starting_index = 1

        for word_index, word in enumerate(text):
            ids = self.huggingface_tokenizer(word, add_special_tokens=False)["input_ids"]
            # if max_len exceeded, stop (leave space for closing token)
            if len(input_ids) + len(ids) >= max_length - 1:
                break
            # token offset before wordpiece, (start, end + 1)
            # offsets.append((len(input_ids), len(input_ids) + len(ids) - 1))
            offsets += [word_index + offset_starting_index] * len(ids)
            words += [word]
            input_ids += ids
            token_type_ids += [token_type_id] * len(ids)
        # last offset
        words += [self.huggingface_tokenizer.sep_token]
        input_ids += [self.huggingface_tokenizer.sep_token_id]
        token_type_ids += [token_type_id]
        offsets.append(len(words) - 1)  # -1 because we want the last index
        return words, input_ids, token_type_ids, offsets

    def pad_batch(self, batch: Union[ModelInputs, Dict[str, list]], max_length: int = None) -> ModelInputs:
        """
        Pad the batch to its maximum length or to the specified :obj:`max_length`.

        Args:
            batch (:obj:`Dict[str, list]`):
                The batch to pad.
            max_length (:obj:`int`, optional):
                Override maximum length of the batch.

        Returns:
            :obj:`Dict[str, list]`: The padded batch.
        """
        if max_length:
            self.subtoken_max_batch_len = max_length
            self.word_max_batch_len = max_length
        else:
            # get maximum len inside a batch
            self.subtoken_max_batch_len = max(len(x) for x in batch["input_ids"])
            self.word_max_batch_len = max(x for x in batch["sentence_lengths"])

        for key in batch:
            if key in self.padding_ops:
                batch[key] = [self.padding_ops[key](b) for b in batch[key]]
        return ModelInputs(batch)

    def pad_sequence(
        self,
        sequence: Union[List, torch.Tensor],
        value: Any = None,
        length: Union[int, str] = "subtoken",
        pad_to_left: bool = False,
    ) -> Union[List, torch.Tensor]:
        """
        Pad the input to the specified length with the given value.

        Args:
            sequence (:obj:`List`, :obj:`torch.Tensor`):
                Element to pad, it can be either a :obj:`List` or a :obj:`torch.Tensor`.
            value (:obj:`Any`, optional):
                Value to use as padding.
            length (:obj:`int`, :obj:`str`, optional, defaults to :obj:`subtoken`):
                Length after pad.
            pad_to_left (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True`, pads to the left, right otherwise.

        Returns:
            :obj:`List`, :obj:`torch.Tensor`: The padded sequence.

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
        if value is None:
            # this is a trick used to pad the offset.
            # here we want the offset pad to be the max offset index in the batch
            # that is the maximum word length in the batch
            value = self.word_max_batch_len - 1
        padding = [value] * abs(length - len(sequence))
        if isinstance(sequence, torch.Tensor):
            if len(sequence.shape) > 1:
                raise ValueError(f"Sequence tensor must be 1D. Current shape is `{len(sequence.shape)}`")
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
            text (:obj:`str`):
                The text to pre-tokenize.
            use_spacy (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True`, uses SpaCy tokenizer.

        Returns:
            :obj:`List[str]`: The pre-tokenized text.
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
            special_tokens_dict (:obj:`Dict`):
                The dictionary containing special tokens. Keys should be in
                the list of predefined special attributes: [``bos_token``, ``eos_token``,
                ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].

        Returns:
            :obj:`int`: Number of tokens added to the vocabulary.

        """
        return self.huggingface_tokenizer.add_special_tokens(special_tokens_dict)

    def add_padding_ops(self, key: str, value: Any, length: Union[int, str]):
        """
        Add padding logic to custom fields.
        If the field is not in :obj:`self.to_tensor_inputs`, this method will add the key to it.

        Args:
            key (:obj:`str`):
                Name of the field in the tokenizer input.
            value (:obj:`Any`):
                Value to use for padding.
            length (:obj:`int`, :obj:`str`):
                Length to pad. It can be an :obj:`int`, or two string value
                - :obj:`subtoken`: the element is padded to the batch max length relative to the subtokens length
                - :obj:`word`: the element is padded to the batch max length relative to the original word length

        Returns:

        """
        if key not in self.to_tensor_inputs:
            self.to_tensor_inputs.add(key)
        self.padding_ops[key] = partial(self.pad_sequence, value=value, length=length)

    def add_to_tensor_inputs(self, names: Union[str, set]):
        """
        Add these keys to the ones that will be converted in Tensors.

        Args:
            names (:obj:`str`, :obj:`set`):
                Name of the field (or fields) to convert to tensors.

        Returns:

        """
        if isinstance(names, str):
            names = {names}
        self.to_tensor_inputs |= names

    def to_tensor(self, batch: Union[ModelInputs, List[dict], dict]) -> ModelInputs:
        """
        Return a the batch in input as Pytorch tensors.
        The fields that are converted in tensors are in :obj:`self.to_tensor_inputs`. By default, only the
        standard model inputs are converted. Use :obj:`self.add_to_tensor_inputs` to add custom fields.

        Args:
            batch (:obj:`List[dict]`, :obj:`dict`):
                Batch in input.

        Returns:
            :obj:`dict`: The batch as tensor.

        """
        # convert to tensor
        batch = {
            k: torch.as_tensor(v) if k in self.to_tensor_inputs and not isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        return ModelInputs(batch)

    def _load_spacy(self) -> "spacy.tokenizer.Tokenizer":
        """
        Download and load spacy model.

        Returns:
            :obj:`spacy.tokenizer.Tokenizer`: The spacy tokenizer.
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
            output (:obj`List[dict]`, :obj:`dict`):
                The output to clean.

        Returns:
            :obj:`dict`: The cleaned output.

        """
        # single sentence case, generalize
        if isinstance(output, dict):
            output = [output]
        # convert list to dict
        output = {k: [d[k] for d in output] for k in output[0]}
        return output

    @staticmethod
    def _get_token_type_id(config: tr.PretrainedConfig) -> int:
        """
        Get token type id. Useful when dealing with models that don't accept 1 as type id.
        Args:
            config (:obj:`transformers.PretrainedConfig`):
                Transformer config.

        Returns:
            :obj:`int`: Correct token type id for that model.

        """
        if hasattr(config, "type_vocab_size"):
            return 1 if config.type_vocab_size == 2 else 0
        return 0

    @staticmethod
    def _type_checking(text: Any, text_pair: Any):
        """
        Checks type of the inputs.

        Args:
            text (:obj:`Any`):
                Text to check.
            text_pair (:obj:`Any`):
                Text pair to check.

        Returns:

        """

        def is_type_correct(text_to_check: Any) -> bool:
            """
            Check if input type is correct, returning a boolean.

            Args:
                text_to_check (:obj:`Any`):
                    text to check.

            Returns:
                :obj`bool`: :obj`True` if the type is correct.

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
                                and (len(text_to_check[0]) == 0 or isinstance(text_to_check[0][0], str))
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
        It assume the input contains both sentences (:obj:`text` and :obj:`text_pair`).

        Returns:
            :obj:`int`: the number of special tokens.

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

    def to(self, device: Union[str, torch.device]) -> ModelInputs:
        """
        Send all tensors values to device.

        Args:
            device (:obj:`str` or :obj:`torch.device`): The device to put the tensors on.

        Returns:
            :class:`tokenizers.ModelInputs`: The same instance of
        :class:`~tokenizers.ModelInputs` after modification.

        """
        if isinstance(device, (str, torch.device, int)):
            self.data = {
                k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in self.data.items()
            }
        else:
            logger.warning(f"Attempting to cast to another type, {str(device)}. This is not supported.")
        return self
