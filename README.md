[![Upload to pypi](https://github.com/Riccorl/transformer-embedder/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Riccorl/transformer-embedder/actions/workflows/python-publish.yml)
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

example = "This is a sample sentence"
inputs = tokenizer(example, return_tensor=True)

# {
#   'input_ids': tensor([[ 101, 1188, 1110,  170, 6876, 5650,  102]]),
#   'offsets': tensor([[[1, 1], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]]),
#   'attention_mask': tensor([[True, True, True, True, True, True, True]]),
#   'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0]])
#   'sentence_length': 7  # with special tokens included
# }

outputs = model(**inputs)

# outputs.shape[1:-1]       # remove [CLS] and [SEP]
# torch.Size([1, 5, 768])
# len(example)
# 5
```

## Info

One of the annoyance of using transfomer-based models is that it is not trivial to compute word embeddings from the sub-token embeddings they output. With this library it's as easy as using 🤗Transformers API to get word-level embeddings from theoretically every transformer model it supports.

### Model

The `TransformerEmbedder` offer 4 ways to retrieve the word embeddings, defined by `subtoken_pooling` parameter:

- `first`: uses only the embedding of the first sub-token of each word
- `last`: uses only the embedding of the last sub-token of each word
- `mean`: computes the mean of the embeddings of the sub-tokens of each word
- `none`: returns the raw output of the transformer model without sub-token pooling

There are also multiple type of outputs you can get using `output_layer` parameter:

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
        fine_tune: bool = True,
    )
```

### Tokenizer

The `Tokenizer` class provides the `tokenize` method to preprocess the input for the `TransformerEmbedder` layer. You
can pass raw sentences, pre-tokenized sentences and sentences in batch. It will preprocess them returning a dictionary
with the inputs for the model. By passing `return_tensor=True` it will return the inputs as `torch.Tensor`.

By default, if you pass text (or batch) as strings, it splits them on spaces
```python
text = "This is a sample sentence"
tokenizer(text)

text = ["This is a sample sentence", "This is another sample sentence"]
tokenizer(text)
```
You can also use SpaCy to pre-tokenize the inputs into words first, using `use_spacy=True`
```python
text = "This is a sample sentence"
tokenizer(text, use_spacy=True)

text = ["This is a sample sentence", "This is another sample sentence"]
tokenizer(text, use_spacy=True)
```

or you can pass an pre-tokenized sentence (or batch of sentences) by setting `is_split_into_words=True`

```python
text = ["This", "is", "a", "sample", "sentence"]
tokenizer(text, is_split_into_words=True)

text = [
    ["This", "is", "a", "sample", "sentence", "1"],
    ["This", "is", "sample", "sentence", "2"],
]
tokenizer(text, is_split_into_words=True) # here is_split_into_words it's redundant
```

Here some examples:

```python
import transformer_embedder as tre

tokenizer = tre.Tokenizer("bert-base-cased")

text = "This is a sample sentence"
tokenizer(text)

# {
#  'input_ids': [101, 1188, 1110, 170, 6876, 5650, 102],
#  'offsets': [(1, 1), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)],
#  'attention_mask': [True, True, True, True, True, True, True],
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0],
#  'sentence_length': 7
#  }

text = "This is a sample sentence A"
text_pair = "This is a sample sentence B"
tokenizer(text, text_pair)

# {
#  'input_ids': [101, 1188, 1110, 170, 6876, 5650, 138, 102, 1188, 1110, 170, 6876, 5650, 139, 102],
#  'offsets': [(1, 1), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14)],
#  'attention_mask': [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
#  'sentence_length': 15
# }

batch = [
    ["This", "is", "a", "sample", "sentence", "1"],
    ["This", "is", "sample", "sentence", "2"],
    ["This", "is", "a", "sample", "sentence", "3"],
    # ...
    ["This", "is", "a", "sample", "sentence", "n", "for", "batch"],
]
tokenizer(batch, padding=True, return_tensor=True)

batch_pair = [
    ["This", "is", "a", "sample", "sentence", "pair", "1"],
    ["This", "is", "sample", "sentence", "pair", "2"],
    ["This", "is", "a", "sample", "sentence", "pair", "3"],
    # ...
    ["This", "is", "a", "sample", "sentence", "pair", "n", "for", "batch"],
]
tokenizer(batch, batch_pair, padding=True, return_tensor=True)
```

## To-Do

Future developments
- [X] Add an optional word tokenizer, maybe using SpaCy

## Acknowledgement

Most of the code in the `TransformerEmbedder` class is taken from the [AllenNLP](https://github.com/allenai/allennlp) 
library. The pretrained models and the core of the tokenizer is from [🤗Transformers](https://huggingface.co/transformers/).
