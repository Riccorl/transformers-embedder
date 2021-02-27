from pathlib import Path
from typing import Optional, Union, List

import torch
import transformers as tr

from transformer_embedder import utils

# Most of the code is taken from [AllenNLP](https://github.com/allenai/allennlp)


logger = utils.get_logger(__name__)
utils.get_logger("transformers")


class TransformerEmbedder(torch.nn.Module):
    """Transforeemer Embedder class."""

    def __init__(
        self,
        model_name: str,
        subtoken_pooling: str = "first",
        output_layer: str = "last",
        fine_tune: bool = True,
    ) -> None:
        """
        Embeddings of words from various transformer architectures from Huggingface Trasnformers API.
        :param model_name: name of the transformer model
        (https://huggingface.co/transformers/pretrained_models.html).
        :param subtoken_pooling: how to get back word embeddings from subtokens. First subtoken (`first`),
        the last subtoken (`last`), or the mean of all the subtokens of the word (`mean`). `none` returns
        the raw output from the transformer model.
        :param output_layer: what output to get from the transformer model. The last hidden state (`last`),
        the concatenation of the last four hidden layers (`concat`), the sum of the last four hidden layers
         (`sum`), the pooled output (`pooled`).
        :param fine_tune: if True, the transformer model is fine-tuned during training.
        """
        super().__init__()
        config = tr.AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.transformer_model = tr.AutoModel.from_pretrained(model_name, config=config)
        self.subtoken_pooling = subtoken_pooling
        self.output_layer = output_layer
        if not fine_tune:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

    @property
    def hidden_size(self) -> int:
        """
        Returns the hidden size of the transformer.
        :return: hidden size of self.transformer_model
        """
        multiplayer = 4 if self.output_layer == "concat" else 1
        return self.transformer_model.config.hidden_size * multiplayer

    def forward(
        self,
        input_ids: torch.LongTensor,
        offsets: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward method of the PyTorch module.
        :param input_ids: Input ids for the transformer model
        :param offsets: Offsets of the sub-token, used to reconstruct the word embeddings
        :param attention_mask: Attention mask for the transformer model
        :param token_type_ids: Token type ids for the transformer model
        :param args:
        :param kwargs:
        :return: the word embeddings
        """
        # Shape: [batch_size, num_subtoken, embedding_size].
        transformer_outputs = self.transformer_model(
            input_ids, attention_mask, token_type_ids
        )
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
                f"`sum`, `pooled`. Current value {self.output_layer}"
            )
        word_embeddings = self.get_word_embeddings(embeddings, offsets)
        return word_embeddings

    def get_word_embeddings(
        self, embeddings: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve the word embeddings from the subtokens embeddings by either computing the
        mean of the subtokens or taking one (first or last) as word representation.
        :param embeddings: subtokens embeddings
        :param offsets: offsets of the subtokens
        :return: the word embeddings
        """
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

    @staticmethod
    def merge_subtoken_embeddings(
        embeddings: torch.Tensor, span_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge subtokens of a word by computing their mean.
        :param embeddings: subtokens embeddings
        :param span_mask: span_mask
        :return: the word embeddings
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
        Get the first or last subtoken as word representation.
        :param embeddings: subtoken embeddings
        :param position: 0 for first subtoken, -1 for last subtoken
        :return: the word embeddings
        """
        return embeddings[:, :, position, :]

    def save_pretrained(self, save_directory: Union[str, Path]):
        """
        Save a model and its configuration file to a directory.
        :param save_directory: Directory to which to save.
        :return:
        """
        self.transformer_model.save_pretrained(save_directory)
