from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple

import torch
import transformers as tr

from transformer_embedder import utils

logger = utils.get_logger(__name__)
utils.get_logger("transformers")


@dataclass
class WordsModelOutput(tr.file_utils.ModelOutput):
    """Class for model's outputs."""

    word_embeddings: torch.Tensor = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class TransformerEmbedder(torch.nn.Module):
    """Transforeemer Embedder class."""

    def __init__(
        self,
        model: Union[str, tr.PreTrainedModel],
        subtoken_pooling: str = "first",
        output_layer: str = "last",
        fine_tune: bool = True,
        return_all: bool = False,
    ) -> None:
        """
        Embeddings of words from various transformer architectures from Huggingface Trasnformers API.

        Args:
            model (`str` or `tr.PreTrainedModel`): transformer model to use
                (https://huggingface.co/transformers/pretrained_models.html).
            subtoken_pooling (): how to get back word embeddings from sub-tokens. First sub-token (`first`),
                the last sub-token (`last`), or the mean of all the sub-tokens of the word (`mean`). `none`
                returns the raw output from the transformer model.
            output_layer (): what output to get from the transformer model. The last hidden state (`last`),
                the concatenation of the last four hidden layers (`concat`), the sum of the last four hidden
                layers (`sum`), the pooled output (`pooled`).
            fine_tune (): if `True`, the transformer model is fine-tuned during training.
            return_all (): if `True`, returns all the outputs from the HuggingFace model.
        """
        super().__init__()
        if isinstance(model, str):
            config = tr.AutoConfig.from_pretrained(
                model, output_hidden_states=True, output_attention=True
            )
            self.transformer_model = tr.AutoModel.from_pretrained(model, config=config)
        else:
            self.transformer_model = model
        self.subtoken_pooling = subtoken_pooling
        self.output_layer = output_layer
        self.return_all = return_all
        if not fine_tune:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor,
        offsets: torch.LongTensor = None,
        attention_mask: torch.BoolTensor = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        *args,
        **kwargs,
    ) -> WordsModelOutput:
        """
        Forward method of the PyTorch module.

        Args:
            input_ids (torch.LongTensor): Input ids for the transformer model
            offsets (torch.LongTensor): Offsets of the sub-token, used to reconstruct the word embeddings
            attention_mask (torch.BoolTensor): Attention mask for the transformer model
            token_type_ids (torch.LongTensor): Token type ids for the transformer model
            *args ():
            **kwargs ():

        Returns:
             the word embeddings

        """
        # Some of the huggingface models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        max_type_id = token_type_ids.max()
        if max_type_id == 0:
            token_type_ids = None
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        # Shape: [batch_size, num_subtoken, embedding_size].
        transformer_outputs = self.transformer_model(**inputs)
        if self.output_layer == "last":
            embeddings = transformer_outputs.last_hidden_state
        elif self.output_layer == "concat":
            embeddings = transformer_outputs.hidden_states[-4:]
            embeddings = torch.cat(embeddings, dim=-1)
        elif self.output_layer == "sum":
            embeddings = transformer_outputs.hidden_states[-4:]
            embeddings = torch.stack(embeddings, dim=0).sum(dim=0)
        elif self.output_layer == "pooled":
            embeddings = transformer_outputs.pooler_output
        else:
            raise ValueError(
                "output_layer parameter not valid, choose between `last`, `concat`, "
                f"`sum`, `pooled`. Current value `{self.output_layer}`"
            )
        word_embeddings = self.get_word_embeddings(embeddings, offsets)
        if self.return_all:
            return WordsModelOutput(
                word_embeddings=word_embeddings,
                last_hidden_state=transformer_outputs.last_hidden_state,
                hidden_states=transformer_outputs.hidden_states,
                pooler_output=transformer_outputs.pooler_output,
                attentions=transformer_outputs.attentions,
            )
        return WordsModelOutput(word_embeddings=word_embeddings)

    def get_word_embeddings(
        self, embeddings: torch.Tensor, offsets: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Retrieve the word embeddings from the sub-tokens embeddings.
        It computes the mean of the sub-tokens or taking one (first or last) as word representation.

        Args:
            embeddings (torch.Tensor): sub-tokens embeddings
            offsets (torch.Tensor): offsets of the sub-tokens

        Returns:
            torch.Tensor: the word embeddings

        """
        # no offsets provided, returns the embeddings as they are.
        # subtoken_pooling parameter ignored.
        if offsets is None:
            return embeddings
        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = utils.batched_span_select(
            embeddings.contiguous(), offsets
        )
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings
        if self.subtoken_pooling == "first":
            word_embeddings = self.single_subtoken_embeddings(span_embeddings, 0)
        elif self.subtoken_pooling == "last":
            word_embeddings = self.single_subtoken_embeddings(span_embeddings, -1)
        elif self.subtoken_pooling == "mean":
            word_embeddings = self.merge_subtoken_embeddings(span_embeddings, span_mask)
        elif self.subtoken_pooling == "none":
            word_embeddings = embeddings
        else:
            raise ValueError(
                f"{self.subtoken_pooling} pooling mode not valid. Choose between "
                "`first`, `last`, `mean` and `none`"
            )
        return word_embeddings

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None
    ) -> torch.nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if :obj:`new_num_tokens != config.vocab_size`.

        Args:
            new_num_tokens (int): The number of new tokens in the embedding matrix.

        Returns:
            torch.nn.Embedding: Pointer to the input tokens Embeddings Module of the model.

        """
        return self.transformer_model.resize_token_embeddings(new_num_tokens)

    @property
    def hidden_size(self) -> int:
        """
        Returns the hidden size of the transformer.

        Returns:
            int: hidden size of self.transformer_model

        """
        multiplier = 4 if self.output_layer == "concat" else 1
        return self.transformer_model.config.hidden_size * multiplier

    @staticmethod
    def merge_subtoken_embeddings(
        embeddings: torch.Tensor, span_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge sub-tokens of a word by computing their mean.
        Most of the code is taken from [AllenNLP](https://github.com/allenai/allennlp).

        Args:
            embeddings (torch.Tensor): sub-tokens embeddings
            span_mask (torch.Tensor): span_mask

        Returns:
            torch.Tensor: the word embeddings

        """
        embeddings_sum = embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        word_embeddings = embeddings_sum / torch.clamp_min(span_embeddings_len, 1)
        # All the places where the span length is zero, write in zeros.
        word_embeddings[(span_embeddings_len == 0).expand(word_embeddings.shape)] = 0
        return word_embeddings

    @staticmethod
    def single_subtoken_embeddings(
        embeddings: torch.Tensor, position: int
    ) -> torch.Tensor:
        """
        Get the first or last sub-token as word representation.

        Args:
            embeddings (torch.Tensor): sub-token embeddings
            position (int): 0 for first sub-token, -1 for last sub-token

        Returns:
            torch.Tensor: the word embeddings

        """
        return embeddings[:, :, position, :]

    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        Save a model and its configuration file to a directory.

        Args:
            save_directory (`str` or `Path`): Directory to which to save

        Returns:

        """
        self.transformer_model.save_pretrained(save_directory)
