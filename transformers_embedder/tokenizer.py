from __future__ import annotations

from collections import UserDict
from functools import partial
from typing import List, Dict, Union, Any, Optional, Tuple, Set

import transformers as tr
from transformers import BatchEncoding
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from transformers_embedder import MODELS_WITH_STARTING_TOKEN, MODELS_WITH_DOUBLE_SEP
from transformers_embedder import utils
from transformers_embedder.utils import is_torch_available

if is_torch_available():
    import torch

logger = utils.get_logger(__name__)
utils.get_logger("transformers")


class Tokenizer:
    """
    A wrapper class for HuggingFace Tokenizer.

    Args:
        model (:obj:`str`, :obj:`transformers.PreTrainedTokenizer`):
            Language model name (or a transformer :obj:`PreTrainedTokenizer`.
    """

    def __init__(self, model: Union[str, tr.PreTrainedTokenizer], *args, **kwargs):
        if isinstance(model, str):
            # init HuggingFace tokenizer
            self.huggingface_tokenizer = tr.AutoTokenizer.from_pretrained(model, *args, **kwargs)
            # get config
            self.config = tr.AutoConfig.from_pretrained(model, *args, **kwargs)
        else:
            self.huggingface_tokenizer = model
            self.config = tr.AutoConfig.from_pretrained(
                self.huggingface_tokenizer.name_or_path, *args, **kwargs
            )
        # padding stuff
        # default, batch length is model max length
        self.subword_max_batch_len = self.huggingface_tokenizer.model_max_length
        self.word_max_batch_len = self.huggingface_tokenizer.model_max_length
        # padding ops
        self.padding_ops = {}
        # keys that will be converted in tensors
        self.to_tensor_inputs = set()

    def __len__(self):
        """Size of the full vocabulary with the added tokens."""
        return len(self.huggingface_tokenizer)

    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]],
        text_pair: Union[str, List[str], List[List[str]], None] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[bool, str]] = None,
        is_split_into_words: bool = False,
        additional_inputs: Optional[Dict[str, Any]] = None,
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
            additional_inputs (:obj:`Dict[str, Any]`, optional, defaults to :obj:`None`):
                Additional inputs to be passed to the model.

        Returns:
            :obj:`ModelInputs`: The inputs to the transformer model.
        """
        # some checks before starting
        if return_tensors is True:
            return_tensors = "pt"
        if return_tensors is False:
            return_tensors = None

        # use huggingface tokenizer to encode the text
        model_inputs = self.huggingface_tokenizer(
            text,
            text_pair=text_pair,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            is_split_into_words=is_split_into_words,
            return_tensors=return_tensors,
            *args,
            **kwargs,
        )
        # build the offsets used to pool the subwords
        offsets, sentence_lengths = self.build_offsets(
            model_inputs, return_tensors=return_tensors, there_is_text_pair=text_pair is not None
        )

        # convert to ModelInputs
        model_inputs = ModelInputs(**model_inputs)
        # add the offsets to the model inputs
        model_inputs.update({"offsets": offsets, "sentence_lengths": sentence_lengths})

        # check if we need to convert other stuff to tensors
        if additional_inputs:
            model_inputs.update(additional_inputs)
            # check if there is a padding strategy
            if padding:
                missing_keys = set(additional_inputs.keys()) - set(self.padding_ops.keys())
                if missing_keys:
                    raise ValueError(
                        f"There are no padding strategy for the following keys: {missing_keys}. "
                        "Please add one with `tokenizer.add_padding_ops()`."
                    )
                self.pad_batch(model_inputs)
            # convert them to tensors
            if return_tensors == "pt":
                self.to_tensor(model_inputs)

        return model_inputs

    def build_offsets(
        self,
        model_inputs: BatchEncoding,
        return_tensors: bool = True,
        there_is_text_pair: bool = False,
    ) -> Tuple:
        """
        Build the offset tensor for the batch of inputs.

        Args:
            model_inputs (:obj:`BatchEncoding`):
                The inputs to the transformer model.
            return_tensors (:obj:`bool`, optional, defaults to :obj:`True`):
                If :obj:`True`, the outputs is converted to :obj:`torch.Tensor`
            there_is_text_pair (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True` `text_pair` is not None.

        Returns:
            :obj:`List[List[int]]` or :obj:`torch.Tensor`: The offsets of the sub-tokens.
        """
        # output data structure
        offsets = []
        sentence_lengths = []
        # this is used as padding value for the offsets
        max_batch_offset = 0
        # model_inputs should be the output of the HuggingFace tokenizer
        # it contains the word offsets to reconstruct the original tokens from the
        # sub-tokens
        for batch_index in range(len(model_inputs.input_ids)):
            word_ids = model_inputs.word_ids(batch_index)
            # it is slightly different from what we need, so here we make it compatible
            # with our subword pooling strategy
            # if the first token is a special token, we need to take it into account
            if self.has_starting_token:
                word_offsets = [0] + [w + 1 if w is not None else w for w in word_ids[1:]]
            # otherwise, we can just use word_ids as is
            else:
                word_offsets = word_ids
            # here we retrieve the max offset for the sample, which will be used as SEP offset
            # and also as padding value for the offsets
            sep_offset = max([w for w in word_offsets if w is not None]) + 1
            # replace first None occurrence with sep_offset
            sep_index = word_offsets.index(None)
            word_offsets[sep_index] = sep_offset
            # if there is a text pair, we need to adjust the offsets for the second text
            if there_is_text_pair:
                # some models have two SEP tokens in between the two texts
                if self.has_double_sep:
                    sep_index += 1
                    sep_offset += 1
                    word_offsets[sep_index] = sep_offset
                # keep the first offsets as is, adjust the second ones
                word_offsets = word_offsets[: sep_index + 1] + [
                    w + sep_offset if w is not None else w for w in word_offsets[sep_index + 1 :]
                ]
                # update again the sep_offset
                sep_offset = max([w for w in word_offsets if w is not None]) + 1
                # replace first None occurrence with sep_offset
                # now it should be the last one
                sep_index = word_offsets.index(None)
                word_offsets[sep_index] = sep_offset
            # keep track of the maximum offset for padding
            max_batch_offset = max(max_batch_offset, sep_offset)
            offsets.append(word_offsets)
            sentence_lengths.append(sep_offset + 1)
        # replace remaining None occurrences with max_batch_offset
        offsets = [[o if o is not None else max_batch_offset for o in offset] for offset in offsets]
        # if return_tensor is True, we need to convert the offsets to tensors
        if return_tensors:
            offsets = torch.as_tensor(offsets)
        return offsets, sentence_lengths

    def pad_batch(
        self, batch: Union[ModelInputs, Dict[str, list]], max_length: Optional[int] = None
    ) -> ModelInputs:
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
            self.subword_max_batch_len = max_length
            self.word_max_batch_len = max_length
        else:
            # get maximum len inside a batch
            self.subword_max_batch_len = max(len(x) for x in batch["input_ids"])
            self.word_max_batch_len = max(x for x in batch["sentence_lengths"])

        for key in batch:
            if key in self.padding_ops:
                batch[key] = [self.padding_ops[key](b) for b in batch[key]]
        return ModelInputs(batch)

    def pad_sequence(
        self,
        sequence: Union[List, torch.Tensor],
        value: Optional[Any] = None,
        length: Union[int, str] = "subword",
        pad_to_left: bool = False,
    ) -> Union[List, torch.Tensor]:
        """
        Pad the input to the specified length with the given value.

        Args:
            sequence (:obj:`List`, :obj:`torch.Tensor`):
                Element to pad, it can be either a :obj:`List` or a :obj:`torch.Tensor`.
            value (:obj:`Any`, optional):
                Value to use as padding.
            length (:obj:`int`, :obj:`str`, optional, defaults to :obj:`subword`):
                Length after pad.
            pad_to_left (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True`, pads to the left, right otherwise.

        Returns:
            :obj:`List`, :obj:`torch.Tensor`: The padded sequence.
        """
        if length == "subword":
            length = self.subword_max_batch_len
        elif length == "word":
            length = self.word_max_batch_len
        else:
            if not isinstance(length, int):
                raise ValueError(
                    f"`length` must be an `int`, `subword` or `word`. Current value is `{length}`"
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
                - ``subword``: the element is padded to the batch max length relative to the subwords length
                - ``word``: the element is padded to the batch max length relative to the original word length
        """
        if key not in self.to_tensor_inputs:
            self.to_tensor_inputs.add(key)
        self.padding_ops[key] = partial(self.pad_sequence, value=value, length=length)

    def add_to_tensor_inputs(self, names: Union[str, set]) -> Set[str]:
        """
        Add these keys to the ones that will be converted in Tensors.

        Args:
            names (:obj:`str`, :obj:`set`):
                Name of the field (or fields) to convert to tensors.

        Returns:
            :obj:`set`: The set of keys that will be converted to tensors.
        """
        if isinstance(names, str):
            names = {names}
        self.to_tensor_inputs |= names
        return self.to_tensor_inputs

    def to_tensor(self, batch: Union[ModelInputs, List[dict], dict]) -> ModelInputs:
        """
        Return the batch in input as Pytorch tensors. The fields that are converted in tensors are in
        :obj:`self.to_tensor_inputs`. By default, only the standard model inputs are converted. Use
        :obj:`self.add_to_tensor_inputs` to add custom fields.

        Args:
            batch (:obj:`List[dict]`, :obj:`dict`):
                Batch in input.

        Returns:
            :obj:`ModelInputs`: The batch as tensor.
        """
        # convert to tensor
        batch = {
            k: torch.as_tensor(v) if k in self.to_tensor_inputs and not isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        return ModelInputs(batch)

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
            Check if input type is correct, returning a boolean value.

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
                "pre-tokenized example) or `List[List[str]]` (batch of pre-tokenized examples)."
            )

        if not is_type_correct(text_pair):
            raise AssertionError(
                "text_pair input must be `str` (single example), `List[str]` (batch or single "
                "pre-tokenized example) or `List[List[str]]` (batch of pre-tokenized examples)."
            )

    @property
    def num_special_tokens(self) -> int:
        """
        Return the number of special tokens the model needs.
        It assumes the input contains both sentences (:obj:`text` and :obj:`text_pair`).

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
            :class:`tokenizers.ModelInputs`: The same instance of :class:`~tokenizers.ModelInputs`
            after modification.
        """
        if isinstance(device, (str, torch.device, int)):
            self.data = {
                k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in self.data.items()
            }
        else:
            logger.warning(f"Attempting to cast to another type, {str(device)}. This is not supported.")
        return self
