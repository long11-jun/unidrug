from abc import ABC, abstractmethod
from transformers import BertModel, BertTokenizer, T5Model, T5Tokenizer, GPT2Model, GPT2Tokenizer

from feat.base_featurizer import BaseFeaturizer
from utils import ToDevice

# Warning: it seems that the results of AutoTokenizer and specified tokenizer is different
name2tokenizer = {
    "bert": BertTokenizer,
    "t5": T5Tokenizer,
    "gpt2": GPT2Tokenizer
}
name2model = {
    "bert": BertModel,
    "t5": T5Model,
    "gpt2": GPT2Model
}


class TextFeaturizer(BaseFeaturizer, ABC):
    def __init__(self):
        super(TextFeaturizer).__init__()
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError


class TextTransformerTokFeaturizer(TextFeaturizer):
    def __init__(self, config):
        super(TextTransformerTokFeaturizer, self).__init__()
        self.max_length = config["max_length"]
        self.fixed_length = config['fixed_length']
        self.tokenizer = name2tokenizer[config["transformer_type"]].from_pretrained(config["model_name_or_path"],
                                                                                    model_max_length=self.max_length)
        if config["transformer_type"] in ["gpt2"]:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, data):
        if self.transform is not None:
            data = self.transform[data]

        if self.fixed_length:
            tokens = self.tokenizer(data, truncation=True, padding='max_length')
        else:
            tokens = self.tokenizer(data, truncation=True, padding=True)
        return tokens


class TextTransformerSentFeaturizer(TextFeaturizer):
    def __init__(self, config):
        super(TextTransformerSentFeaturizer, self).__init__()
        self.max_length = config["max_length"]
        self.min_sentence_length = config["min_sentence_length"]
        self.tokenizer = name2tokenizer[config["transformer_type"]].from_pretrained(config["model_name_or_path"],
                                                                                    model_max_length=self.max_length)
        if config["transformer_type"] in ["gpt2"]:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, data):
        if self.transform is not None:
            data = self.transform[data]
        sents = []
        for sent in data.split("."):
            if len(sent.split(" ")) < 5:
                continue
            sents.append(self.tokenizer(sent, truncation=True, padding=True))
        return sents


class TextTransformerEncFeaturizer(TextFeaturizer):
    def __init__(self, config):
        super(TextTransformerEncFeaturizer, self).__init__()
        self.device = config["device"]
        self.max_length = config["max_length"]
        self.fixed_length = config['fixed_length']
        self.output_type = config['output_type']
        self.tokenizer = name2tokenizer[config["transformer_type"]].from_pretrained(config["model_name_or_path"],
                                                                                    model_max_length=self.max_length)
        if config["transformer_type"] in ["gpt2"]:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = name2model[config["transformer_type"]].from_pretrained(config["model_name_or_path"])
        self.encoder = self.encoder.to(self.device)

    def __call__(self, data):
        if self.transform is not None:
            data = self.transform[data]

        if self.fixed_length:
            data = self.tokenizer(data, truncation=True, padding='max_length', return_tensors='pt')
        else:
            data = self.tokenizer(data, truncation=True, padding=True, return_tensors='pt')
        data = ToDevice(data, self.device)
        data = self.encoder(**data)[self.output_type].detach().cpu()
        return data


class RawStringFeaturizer(TextFeaturizer):
    """Return raw string (or list of strings) without tokenization."""

    def __init__(self, config):
        super(RawStringFeaturizer, self).__init__()
        self.max_length = config.get("max_length")

    def __call__(self, data):
        if self.transform is not None:
            data = self.transform[data]

        if data is None:
            return ""

        if isinstance(data, str):
            return data[: self.max_length] if self.max_length else data
        if isinstance(data, list):
            if self.max_length:
                return [d[: self.max_length] if isinstance(d, str) else d for d in data]
            return data
        return str(data)


SUPPORTED_TEXT_FEATURIZER = {
    "TransformerTokenizer": TextTransformerTokFeaturizer,
    "TransformerSentenceTokenizer": TextTransformerSentFeaturizer,
    "TransformerEncoder": TextTransformerEncFeaturizer,
    "RawString": RawStringFeaturizer,
}