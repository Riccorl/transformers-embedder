![Upload Python Package](https://github.com/Riccorl/transfomrer-embedder/workflows/Upload%20Python%20Package/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Transformer Embedder

A Word Level Transformer layer based on Pytorch and 🤗Transformers. 

## How to use

Install the library

```bash
pip install transformer-embedder
```

It offers a Pytorch layer and a tokenizer that support almost every pretrained model from Huggingface
[🤗Transformers](https://huggingface.co/transformers/) library. Here is a quick example:

```python
import transformer_embedder as tre

model = tre.TransformerEmbedder("bert-base-cased", subtoken_pooling="mean", output_layer="sum")
tokenizer = tre.Tokenizer("bert-base-cased")

example = "This is a sample sentence".split(" ")
inputs = tokenizer(example, return_tensor=True)
"""
{
    'input_ids': tensor([[ 101, 1188, 1110,  170, 6876, 5650,  102]]),
    'offsets': tensor([[[1, 1], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]]),
    'attention_mask': tensor([[True, True, True, True, True, True, True]]),
    'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0]])
}
"""

outputs = model(**inputs)

# outputs.shape[1:-1]       # remove [CLS] and [SEP]
# torch.Size([1, 5, 768])
# len(example)
# 5
```

## Info

One of the annoyance of using transfomer-based models is that is not trivial to compute word embeddings
from the sub-token embeddings that they output. With this library it's as easy as using 🤗Transformers API to get word-level
embeddings from theoretically every transformer model supported by it.


### Model

The `TransformerEmbedder` offer 3 ways to retrieve the word embeddings:

- `first`: uses only the embedding of the first sub-token of each word
- `last`: uses only the embedding of the last sub-token of each word
- `mean`: computes the mean of the embeddings of the sub-tokens of each word

There are also multiple type of outputs:

- `last`: returns the last hidden state of the transformer model
- `concat`: returns the concatenation of the last four hidden states of the transformer model
- `sum`: returns the sum of the last four hidden states of the transformer model
- `pooled`: returns the output of the pooling layer

```python
class TransformerEmbedder(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        subtoken_pooling: str = "first",
        output_layer: str = "last",
        fine_tune: bool = False,
    )
```

### Tokenizer

TO-DO

## Acknowledgement

Most of the code in the `TransformerEmbedder` class is taken from the [AllenNLP](https://github.com/allenai/allennlp) 
library. The pretrained models and the core of the tokenizer is from [🤗Transformers](https://huggingface.co/transformers/).