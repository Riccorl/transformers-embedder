from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple

import transformers as tr

from transformers_embedder import utils

if utils.is_torch_available():
    import torch

logger = utils.get_logger(__name__)
utils.get_logger("transformers")


@dataclass
class TransformersEmbedderOutput(tr.file_utils.ModelOutput):
    """Class for model's outputs."""

    word_embeddings: "torch.Tensor" = None
    last_hidden_state: "torch.FloatTensor" = None
    pooler_output: "torch.FloatTensor" = None
    hidden_states: Optional[Tuple["torch.FloatTensor"]] = None
    attentions: Optional[Tuple["torch.FloatTensor"]] = None


class TransformersEmbedder(torch.nn.Module):
    """
    Transformer Embedder class.
    Word level embeddings from various transformer architectures from Huggingface Trasnformers API.

    Args:
        model (:obj:`str`, :obj:`tr.PreTrainedModel`):
            Transformer model to use (https://huggingface.co/models).
        return_words (:obj:`bool`, optional, defaults to :obj:`True`):
            If ``True`` it returns the word-level embeddings by computing the mean of the
            sub-words embeddings.
        output_layer (:obj:`str`, optional, defaults to :obj:`last`):
            What output to get from the transformer model. The last hidden state (``last``),
            the concatenation of the last four hidden layers (``concat``), the sum of the last four hidden
            layers (``sum``), the pooled output (``pooled``).
        fine_tune (:obj:`bool`, optional, defaults to :obj:`True`):
            If ``True``, the transformer model is fine-tuned during training.
        return_all (:obj:`bool`, optional, defaults to :obj:`False`):
            If ``True``, returns all the outputs from the HuggingFace model.
    """

    def __init__(
        self,
        model: Union[str, tr.PreTrainedModel],
        return_words: bool = True,
        output_layer: str = "last",
        fine_tune: bool = True,
        return_all: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(model, str):
            config = tr.AutoConfig.from_pretrained(model, output_hidden_states=True, output_attentions=True)
            self.transformer_model = tr.AutoModel.from_pretrained(model, config=config)
        else:
            self.transformer_model = model
        self.return_words = return_words
        self.output_layer = output_layer
        self.return_all = return_all
        if not fine_tune:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        offsets: torch.LongTensor = None,
        *args,
        **kwargs,
    ) -> TransformersEmbedderOutput:
        """
        Forward method of the PyTorch module.

        Args:
            input_ids (:obj:`torch.Tensor`, optional):
                Input ids for the transformer model.
            attention_mask (:obj:`torch.Tensor`, optional):
                Attention mask for the transformer model.
            token_type_ids (:obj:`torch.Tensor`, optional):
                Token type ids for the transformer model.
            offsets (:obj:`torch.Tensor`, optional):
                Offsets of the sub-token, used to reconstruct the word embeddings.

        Returns:
             :obj:`TransformersEmbedderOutput`:
                Word level embeddings plus the output of the transformer model.
        """
        # Some of the HuggingFace models don't have the
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
            word_embeddings = transformer_outputs.last_hidden_state
        elif self.output_layer == "concat":
            word_embeddings = transformer_outputs.hidden_states[-4:]
            word_embeddings = torch.cat(word_embeddings, dim=-1)
        elif self.output_layer == "sum":
            word_embeddings = transformer_outputs.hidden_states[-4:]
            word_embeddings = torch.stack(word_embeddings, dim=0).sum(dim=0)
        elif self.output_layer == "pooled":
            word_embeddings = transformer_outputs.pooler_output
        else:
            raise ValueError(
                "output_layer parameter not valid, choose between `last`, `concat`, "
                f"`sum`, `pooled`. Current value `{self.output_layer}`"
            )

        if self.return_words and offsets is None:
            raise ValueError(
                "`return_words` is `True` but `offsets` was not passed to the model. "
                "Cannot compute word embeddings. To solve:\n"
                "- Set `return_words` to `False` or"
                "- Pass `offsets` to the model during forward."
            )
        if self.return_words:
            # Shape: [batch_size, num_token, embedding_size].
            word_embeddings = self.merge_subword(word_embeddings, offsets)
        if self.return_all:
            return TransformersEmbedderOutput(
                word_embeddings=word_embeddings,
                last_hidden_state=transformer_outputs.last_hidden_state,
                hidden_states=transformer_outputs.hidden_states,
                pooler_output=transformer_outputs.pooler_output,
                attentions=transformer_outputs.attentions,
            )
        return TransformersEmbedderOutput(word_embeddings=word_embeddings)

    def merge_subword(
        self,
        embeddings: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Minimal version of ``scatter_mean``, from `pytorch_scatter <https://github.com/rusty1s/pytorch_scatter/>`_
        library, that is compatible for ONNX but works only for our case. It is used to compute word level
        embeddings from the transformer output.

        Args:
            embeddings (:obj:`torch.Tensor`):
                The embeddings tensor.
            indices (:obj:`torch.Tensor`):
                The subword indices.

        Returns:
            :obj:`torch.Tensor`

        """
        out = self.scatter_sum(embeddings, indices)
        ones = torch.ones(indices.size(), dtype=embeddings.dtype, device=embeddings.device)
        count = self.scatter_sum(ones, indices)
        count.clamp_(1)
        count = self.broadcast(count, out)
        out.true_divide_(count)
        return out

    def scatter_sum(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Minimal version of ``scatter_sum``, from `pytorch_scatter <https://github.com/rusty1s/pytorch_scatter/>`_
        library, that is compatible for ONNX but works only for our case.

        Args:
            src (:obj:`torch.Tensor`):
                The source tensor.
            index (:obj:`torch.Tensor`):
                The indices of elements to scatter.

        Returns:
            :obj:`torch.Tensor`

        """
        index = self.broadcast(index, src)
        size = list(src.size())
        size[1] = index.max() + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(1, index, src)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> torch.nn.Embedding:
        """
        Resizes input token embeddings' matrix of the model if :obj:`new_num_tokens != config.vocab_size`.

        Args:
            new_num_tokens (:obj:`int`):
                The number of new tokens in the embedding matrix.

        Returns:
            :obj:`torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.

        """
        return self.transformer_model.resize_token_embeddings(new_num_tokens)

    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        Save a model and its configuration file to a directory.

        Args:
            save_directory (:obj:`str`, :obj:`Path`):
                Directory to which to save.

        Returns:

        """
        self.transformer_model.save_pretrained(save_directory)

    @staticmethod
    def broadcast(src: torch.Tensor, other: torch.Tensor):
        """
        Minimal version of ``broadcast``, from `pytorch_scatter <https://github.com/rusty1s/pytorch_scatter/>`_
        library, that is compatible with ONNX but works only for our case.

        Args:
            src (:obj:`torch.Tensor`):
                The source tensor.
            other (:obj:`torch.Tensor`):
                The tensor from which we want to broadcast.

        Returns:
            :obj:`torch.Tensor`

        """
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand_as(other)
        return src

    @property
    def hidden_size(self) -> int:
        """
        Returns the hidden size of the transformer.

        Returns:
            :obj:`int`: Hidden size of ``self.transformer_model``.

        """
        multiplier = 4 if self.output_layer == "concat" else 1
        return self.transformer_model.config.hidden_size * multiplier
