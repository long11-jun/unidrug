"""

"""
import os
from typing import List, Dict, Union, Optional
import torch


class TokenMolTokenizer:
    """
    
    - pad_token_id: 0 ([PAD])
    - bos_token_id: 825 (<|beginoftext|>)
    - eos_token_id: 826 (<|endofmask|>)
    """
    
    def __init__(
        self,
        vocab_file: str,
        pad_token: str = "[PAD]",
        bos_token: str = "<|beginoftext|>",
        eos_token: str = "<|endofmask|>",
        unk_token: str = "[UNK]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        self.vocab_file = vocab_file
        
        self.vocab: Dict[str, int] = {}
        self.ids_to_tokens: Dict[int, str] = {}
        self._load_vocab(vocab_file)
        
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        
        self.pad_token_id = self.vocab.get(pad_token, 0)
        self.bos_token_id = self.vocab.get(bos_token, self.vocab.get("[CLS]", 0))
        self.eos_token_id = self.vocab.get(eos_token, self.vocab.get("[SEP]", 0))
        self.unk_token_id = self.vocab.get(unk_token, 0)
        self.mask_token_id = self.vocab.get(mask_token, self.vocab.get("<|mask:0|>", 0))
        
        self.vocab_size = len(self.vocab)
        
    def _load_vocab(self, vocab_file: str):
        
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
        
        with open(vocab_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                if token:
                    self.vocab[token] = idx
                    self.ids_to_tokens[idx] = token
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        **kwargs
    ) -> List[int]:
        """
        
        "C C O [qed] 0.85 [logp] 1.2 ..."
        """
        tokens = text.split()
        
        ids = []
        if add_special_tokens and self.bos_token_id is not None:
            ids.append(self.bos_token_id)
        
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.unk_token_id)
        
        if add_special_tokens and self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        
        return ids

    def encode_tokens(
        self,
        tokens: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        
        ids: List[int] = []
        if add_special_tokens and self.bos_token_id is not None:
            ids.append(self.bos_token_id)
        for t in tokens:
            ids.append(self.vocab.get(t, self.unk_token_id))
        if add_special_tokens and self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        
        tokens = []
        for idx in token_ids:
            if skip_special_tokens and idx in special_ids:
                continue
            token = self.ids_to_tokens.get(idx, self.unk_token)
            tokens.append(token)
        
        return " ".join(tokens)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Union[List, torch.Tensor]]:
        """
        
        """
        if isinstance(text, str):
            text = [text]
        
        all_ids = []
        for t in text:
            ids = self.encode(t, add_special_tokens=True, max_length=max_length, truncation=truncation)
            all_ids.append(ids)
        
        # Padding
        if padding:
            max_len = max(len(ids) for ids in all_ids)
            if max_length is not None:
                max_len = min(max_len, max_length)
            
            padded_ids = []
            attention_masks = []
            for ids in all_ids:
                pad_len = max_len - len(ids)
                if pad_len > 0:
                    padded_ids.append(ids + [self.pad_token_id] * pad_len)
                    attention_masks.append([1] * len(ids) + [0] * pad_len)
                else:
                    padded_ids.append(ids[:max_len])
                    attention_masks.append([1] * max_len)
            all_ids = padded_ids
        else:
            attention_masks = [[1] * len(ids) for ids in all_ids]
        
        result = {
            "input_ids": all_ids,
            "attention_mask": attention_masks,
        }
        
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long)
            result["attention_mask"] = torch.tensor(result["attention_mask"], dtype=torch.long)
        
        return BatchEncoding(result)
    
    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()
    
    def __len__(self) -> int:
        return self.vocab_size


class BatchEncoding(dict):
    
    
    def to(self, device):
        result = {}
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(device)
            else:
                result[k] = v
        return BatchEncoding(result)

