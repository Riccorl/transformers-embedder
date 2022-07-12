from typing import Optional

import torch


class Encoder(torch.nn.Module):
    """
    An encoder module for the `TransformersEmbedder` class.

    Args:
        transformer_hidden_size (`int`):
            The hidden size of the inner transformer.
        projection_size (`int`, `optional`, defaults to `None`):
            The size of the projection layer.
        activation_layer (`torch.nn.Module`, optional, defaults to `None`):
            Activation layer to use. If ``None``, no activation layer is used.
        dropout (`float`, `optional`, defaults to `0.1`):
            The dropout value.
        bias (`bool`, `optional`, defaults to `True`):
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
            x (`torch.Tensor`):
                The input tensor.

        Returns:
            `torch.Tensor`: The encoded tensor.
        """
        x = self.projection_layer(self.dropout_layer(x))
        if self.activation_layer is not None:
            x = self.activation_layer(x)
        return x
