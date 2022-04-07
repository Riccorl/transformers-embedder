from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Sequence, Any, Mapping

import transformers as tr
from torch.nn.utils.rnn import pad_sequence

from transformers_embedder import utils

if utils.is_torch_available():
    import torch

logger = utils.get_logger(__name__)
utils.get_logger("transformers")


@dataclass
class TransformersEmbedderOutput(tr.file_utils.ModelOutput):
    """Class for model's outputs."""

    word_embeddings: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class TransformersEmbedder(torch.nn.Module):
    """
    Transformer Embedder class.

    Word level embeddings from various transformer architectures from Huggingface Transformers API.

    Args:
        model (:obj:`str`, :obj:`tr.PreTrainedModel`):
            Transformer model to use (https://huggingface.co/models).
        layer_pooling_strategy (:obj:`str`, optional, defaults to :obj:`last`):
            What output to get from the transformer model. The last hidden state (``last``),
            the concatenation of the selected hidden layers (``concat``), the sum of the selected hidden
            layers (``sum``), the average of the selected hidden layers (``mean``).
        subword_pooling_strategy (:obj:`str`, optional, defaults to :obj:`scatter`):
            What pooling strategy to use for the sub-word embeddings. Methods available are ``scatter``,
            ``sparse`` and ``none``. The ``scatter`` strategy is ONNX comptabile but uses ``scatter_add_``
            that is not deterministic. The ``sparse`` strategy is deterministic but it is not comptabile
            with ONNX. When ``subword_pooling_strategy`` is ``none``, the sub-word embeddings are not
            pooled.
        output_layers (:obj:`tuple`, optional, defaults to :obj:`(-4, -3, -2, -1)`):
            Which hidden layers to get from the transformer model.
        fine_tune (:obj:`bool`, optional, defaults to :obj:`True`):
            If ``True``, the transformer model is fine-tuned during training.
        return_all (:obj:`bool`, optional, defaults to :obj:`False`):
            If ``True``, returns all the outputs from the HuggingFace model.
    """

    def __init__(
        self,
        model: Union[str, tr.PreTrainedModel],
        layer_pooling_strategy: str = "last",
        subword_pooling_strategy: str = "scatter",
        output_layers: Sequence[int] = (-4, -3, -2, -1),
        fine_tune: bool = True,
        return_all: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(model, str):
            config = tr.AutoConfig.from_pretrained(
                model, output_hidden_states=True, output_attentions=True, *args, **kwargs
            )
            self.transformer_model = tr.AutoModel.from_pretrained(model, config=config, *args, **kwargs)
        else:
            self.transformer_model = model
        self.layer_pooling_strategy = layer_pooling_strategy
        self.subword_pooling_strategy = subword_pooling_strategy
        if max(map(abs, output_layers)) >= self.transformer_model.config.num_hidden_layers:
            raise ValueError(
                f"`output_layers` parameter not valid, choose between 0 and "
                f"{self.transformer_model.config.num_hidden_layers - 1}. "
                f"Current value is `{output_layers}`"
            )
        self.output_layers = output_layers
        self.return_all = return_all
        if not fine_tune:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        scatter_offsets: Optional[torch.Tensor] = None,
        sparse_offsets: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> TransformersEmbedderOutput:
        """
        Forward method of the PyTorch module.

        Args:
            input_ids (:obj:`torch.Tensor`):
                Input ids for the transformer model.
            attention_mask (:obj:`torch.Tensor`, optional):
                Attention mask for the transformer model.
            token_type_ids (:obj:`torch.Tensor`, optional):
                Token type ids for the transformer model.
            scatter_offsets (:obj:`torch.Tensor`, optional):
                Offsets of the sub-word, used to reconstruct the word embeddings using the ``scatter`` method.
            sparse_offsets (:obj:`Mapping[str, Any]`, optional):
                Offsets of the sub-word, used to reconstruct the word embeddings using the ``sparse`` method.

        Returns:
             :obj:`TransformersEmbedderOutput`:
                Word level embeddings plus the output of the transformer model.
        """
        # Some HuggingFace models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        # Shape: [batch_size, num_sub-words, embedding_size].
        transformer_outputs = self.transformer_model(**inputs)
        if self.layer_pooling_strategy == "last":
            word_embeddings = transformer_outputs.last_hidden_state
        elif self.layer_pooling_strategy == "concat":
            word_embeddings = [transformer_outputs.hidden_states[layer] for layer in self.output_layers]
            word_embeddings = torch.cat(word_embeddings, dim=-1)
        elif self.layer_pooling_strategy == "sum":
            word_embeddings = [transformer_outputs.hidden_states[layer] for layer in self.output_layers]
            word_embeddings = torch.stack(word_embeddings, dim=0).sum(dim=0)
        elif self.layer_pooling_strategy == "mean":
            word_embeddings = [transformer_outputs.hidden_states[layer] for layer in self.output_layers]
            word_embeddings = torch.stack(word_embeddings, dim=0).mean(dim=0, dtype=torch.float)
        else:
            raise ValueError(
                "`pooling_strategy` parameter not valid, choose between `last`, `concat`, "
                f"`sum` and `mean`. Current value `{self.layer_pooling_strategy}`"
            )

        if self.subword_pooling_strategy != "none" and scatter_offsets is None and sparse_offsets is None:
            raise ValueError(
                "`subword_pooling_strategy` is not `none` but neither `scatter_offsets` not `sparse_offsets` "
                "were passed to the model. Cannot compute word embeddings.\nTo solve:\n"
                "- Set `subword_pooling_strategy` to `none` or\n"
                "- Pass `scatter_offsets` to the model during forward or\n"
                "- Pass `sparse_offsets` to the model during forward."
            )
        if self.subword_pooling_strategy != "none":
            # Shape: [batch_size, num_words, embedding_size].
            if self.subword_pooling_strategy == "scatter":
                if scatter_offsets is None:
                    raise ValueError(
                        "`subword_pooling_strategy` is `scatter` but `scatter_offsets` "
                        "were not passed to the model. Cannot compute word embeddings.\nTo solve:\n"
                        "- Set `subword_pooling_strategy` to `none` or\n"
                        "- Pass `scatter_offsets` to the model during forward."
                    )
                word_embeddings = self.merge_scatter(word_embeddings, indices=scatter_offsets)
            elif self.subword_pooling_strategy == "sparse":
                if sparse_offsets is None:
                    raise ValueError(
                        "`subword_pooling_strategy` is `sparse` but `sparse_offsets` "
                        "were not passed to the model. Cannot compute word embeddings.\nTo solve:\n"
                        "- Set `subword_pooling_strategy` to `none` or\n"
                        "- Pass `sparse_offsets` to the model during forward."
                    )
                word_embeddings = self.merge_sparse(word_embeddings, sparse_offsets)
            else:
                raise ValueError(
                    "`subword_pooling_strategy` parameter not valid, choose between `scatter`, `sparse`"
                    " and `none`."
                    f"Current value is `{self.subword_pooling_strategy}`."
                )

        if self.return_all:
            return TransformersEmbedderOutput(
                word_embeddings=word_embeddings,
                last_hidden_state=transformer_outputs.last_hidden_state,
                hidden_states=transformer_outputs.hidden_states,
                pooler_output=transformer_outputs.pooler_output,
                attentions=transformer_outputs.attentions,
            )
        return TransformersEmbedderOutput(word_embeddings=word_embeddings)

    def merge_scatter(self, embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Minimal version of ``scatter_mean``, from `pytorch_scatter
        <https://github.com/rusty1s/pytorch_scatter/>`_
        library, that is compatible for ONNX but works only for our case. It is used to compute word level
        embeddings from the transformer output.

        Args:
            embeddings (:obj:`torch.Tensor`):
                The embeddings tensor.
            indices (:obj:`torch.Tensor`):
                The sub-word indices.

        Returns:
            :obj:`torch.Tensor`
        """

        def broadcast(src: torch.Tensor, other: torch.Tensor):
            for _ in range(src.dim(), other.dim()):
                src = src.unsqueeze(-1)
            src = src.expand_as(other)
            return src

        def scatter_sum(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            index = broadcast(index, src)
            size = list(src.size())
            size[1] = index.max() + 1
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
            return out.scatter_add_(1, index, src)

        # replace padding indices with the maximum value inside the batch
        indices[indices == -1] = torch.max(indices)
        merged = scatter_sum(embeddings, indices)
        ones = torch.ones(indices.size(), dtype=embeddings.dtype, device=embeddings.device)
        count = scatter_sum(ones, indices)
        count.clamp_(1)
        count = broadcast(count, merged)
        merged.true_divide_(count)
        return merged

    def merge_sparse(self, embeddings: torch.Tensor, bpe_info: Optional[Mapping[str, Any]]) -> torch.Tensor:
        # it is constructed here and not in the tokenizer/collate because pin_memory is not sparse-compatible
        bpe_weights = torch.sparse_coo_tensor(
            indices=bpe_info["sparse_indices"], values=bpe_info["sparse_values"], size=bpe_info["sparse_size"]
        )
        # (sentence, word, bpe) x (sentence, bpe, transformer_dim) -> (sentence, word, transformer_dim)
        merged = torch.bmm(bpe_weights, embeddings)
        return merged

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
        """
        self.transformer_model.save_pretrained(save_directory)

    @property
    def hidden_size(self) -> int:
        """
        Returns the hidden size of TransformersEmbedder.

        Returns:
            :obj:`int`: Hidden size of ``self.transformer_model``.
        """
        multiplier = len(self.output_layers) if self.layer_pooling_strategy == "concat" else 1
        return self.transformer_model.config.hidden_size * multiplier

    @property
    def transformer_hidden_size(self) -> int:
        """
        Returns the hidden size of the inner transformer.

        Returns:
            :obj:`int`: Hidden size of ``self.transformer_model``.
        """
        multiplier = len(self.output_layers) if self.layer_pooling_strategy == "concat" else 1
        return self.transformer_model.config.hidden_size * multiplier


class Encoder(torch.nn.Module):
    """
    An encoder module for the :obj:`TransformersEmbedder` class.

    Args:
        transformer_hidden_size (:obj:`int`):
            The hidden size of the inner transformer.
        projection_size (:obj:`int`, `optional`, defaults to :obj:`None`):
            The size of the projection layer.
        dropout (:obj:`float`, `optional`, defaults to :obj:`0.1`):
            The dropout value.
        bias (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to use a bias.
    """

    def __init__(
        self,
        transformer_hidden_size: int,
        projection_size: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.projection_size = projection_size or transformer_hidden_size
        self.projection_layer = torch.nn.Linear(transformer_hidden_size, self.projection_size, bias=bias)
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.activation_layer = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (:obj:`torch.Tensor`):
                The input tensor.

        Returns:
            :obj:`torch.Tensor`: The encoded tensor.
        """
        return self.activation_layer(self.projection_layer(self.dropout_layer(x)))


class TransformersEncoder(TransformersEmbedder):
    """
    Transformer Embedder class.

    Word level embeddings from various transformer architectures from Huggingface Transformers API.

    Args:
        model (:obj:`str`, :obj:`tr.PreTrainedModel`):
            Transformer model to use (https://huggingface.co/models).
        layer_pooling_strategy (:obj:`str`, optional, defaults to :obj:`last`):
            What output to get from the transformer model. The last hidden state (``last``),
            the concatenation of the selected hidden layers (``concat``), the sum of the selected hidden
            layers (``sum``), the average of the selected hidden layers (``mean``).
        subword_pooling_strategy (:obj:`str`, optional, defaults to :obj:`scatter`):
            What pooling strategy to use for the sub-word embeddings. Methods available are ``scatter``,
            ``sparse`` and ``none``. The ``scatter`` strategy is ONNX comptabile but uses ``scatter_add``
            that is not deterministic. The ``sparse`` strategy is deterministic but it is not comptabile
            with ONNX.
        output_layers (:obj:`tuple`, optional, defaults to :obj:`(-4, -3, -2, -1)`):
            Which hidden layers to get from the transformer model.
        fine_tune (:obj:`bool`, optional, defaults to :obj:`True`):
            If ``True``, the transformer model is fine-tuned during training.
        return_all (:obj:`bool`, optional, defaults to :obj:`False`):
            If ``True``, returns all the outputs from the HuggingFace model.
        projection_size (:obj:`int`, optional, defaults to :obj:`None`):
            If not ``None``, the output of the transformer is projected to this size.
        dropout (:obj:`float`, optional, defaults to :obj:`0.1`):
            The dropout probability.
        bias (:obj:`bool`, optional, defaults to :obj:`True`):
            If ``True``, the transformer model has a bias.
    """

    def __init__(
        self,
        model: Union[str, tr.PreTrainedModel],
        layer_pooling_strategy: str = "last",
        subword_pooling_strategy: str = "scatter",
        output_layers: Sequence[int] = (-4, -3, -2, -1),
        fine_tune: bool = True,
        return_all: bool = False,
        projection_size: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            layer_pooling_strategy,
            subword_pooling_strategy,
            output_layers,
            fine_tune,
            return_all,
            *args,
            **kwargs,
        )
        self.encoder = Encoder(self.transformer_hidden_size, projection_size, dropout, bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        scatter_offsets: Optional[torch.Tensor] = None,
        sparse_offsets: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> TransformersEmbedderOutput:
        """
        Forward method of the PyTorch module.

        Args:
            input_ids (:obj:`torch.Tensor`):
                Input ids for the transformer model.
            attention_mask (:obj:`torch.Tensor`, optional):
                Attention mask for the transformer model.
            token_type_ids (:obj:`torch.Tensor`, optional):
                Token type ids for the transformer model.
            scatter_offsets (:obj:`torch.Tensor`, optional):
                Offsets of the sub-word, used to reconstruct the word embeddings.

        Returns:
             :obj:`TransformersEmbedderOutput`:
                Word level embeddings plus the output of the transformer model.
        """
        transformers_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "scatter_offsets": scatter_offsets,
            "sparse_offsets": sparse_offsets,
            **kwargs,
        }
        transformer_output = super().forward(**transformers_kwargs)
        encoder_output = self.encoder(transformer_output.word_embeddings)
        transformer_output.word_embeddings = encoder_output
        return transformer_output

    @property
    def hidden_size(self) -> int:
        """
        Returns the hidden size of the transformer.

        Returns:
            :obj:`int`: Hidden size of ``self.transformer_model``.
        """
        return self.encoder.projection_size
