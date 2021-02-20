from transformers import (
    RobertaTokenizerFast,
    XLMRobertaTokenizerFast,
    BertTokenizerFast,
    DistilBertTokenizerFast,
    MobileBertTokenizerFast,
    BertweetTokenizer,
    CamembertTokenizerFast,
    DebertaTokenizer,
    XLMTokenizer,
)

MODELS_WITH_STARTING_TOKEN = (
    BertTokenizerFast,
    DistilBertTokenizerFast,
    MobileBertTokenizerFast,
    BertweetTokenizer,
    CamembertTokenizerFast,
    DebertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizerFast,
    XLMTokenizer,
)

MODELS_WITH_DOUBLE_SEP = (
    CamembertTokenizerFast,
    BertweetTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizerFast,
)

from transformer_embedder.embedder import TransformerEmbedder
from transformer_embedder.tokenizer import Tokenizer
