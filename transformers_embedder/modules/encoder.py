from typing import Optional

import torch


class Encoder(torch.nn.Module):
    """
    An encoder module for the :obj:`TransformersEmbedder` class.

    Args:
        transformer_hidden_size (:obj:`int`):
            The hidden size of the inner transformer.
        projection_size (:obj:`int`, `optional`, defaults to :obj:`None`):
            The size of the projection layer.
        activation_layer (:obj:`torch.nn.Module`, optional, defaults to :obj:`None`):
            Activation layer to use. If ``None``, no activation layer is used.
        dropout (:obj:`float`, `optional`, defaults to :obj:`0.1`):
            The dropout value.
        bias (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to use a bias.
    """

    def __init__(
        self,
        transformer_hidden_size: int,
        projection_size: Optional[int] = None,
        activation_layer: Optional[torch.nn.Module] = None,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.projection_size = projection_size or transformer_hidden_size
        self.projection_layer = torch.nn.Linear(
            transformer_hidden_size, self.projection_size, bias=bias
        )
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.activation_layer = activation_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (:obj:`torch.Tensor`):
                The input tensor.

        Returns:
            :obj:`torch.Tensor`: The encoded tensor.
        """
        x = self.projection_layer(self.dropout_layer(x))
        if self.activation_layer is not None:
            x = self.activation_layer(x)
        return x
