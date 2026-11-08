import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..base_models import TextEncoder

class QwenEncoder(TextEncoder):
    """
    
    """
    def __init__(self, config):
        super(QwenEncoder, self).__init__()
        self.model_name = config.get("model_name", "Qwen/Qwen2-0.5B")
        self.max_length = config.get("max_length", 512)
        self.device = config.get("device", "cuda:0")
        
        dtype_str = str(config.get("torch_dtype", "bfloat16")).lower()
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)
        
        import os
        is_local = os.path.exists(self.model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=is_local
        )
        

        print(f"Loading Qwen: {self.model_name}")
        print(f"   Device: {self.device}")
        
        if self.device == "cpu":
            print("Loading on CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=is_local
            )
            print("Loaded on CPU")
        else:
            if torch.cuda.is_available():
                gpu_id = int(self.device.split(':')[1]) if ':' in self.device else 0
                if gpu_id >= torch.cuda.device_count():
                    print(f"GPU {gpu_id} is not available, loading on CPU")
                    self.device = "cpu"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        local_files_only=is_local
                    )
                else:
                    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                    allocated_memory = torch.cuda.memory_allocated(gpu_id)
                    free_memory = total_memory - allocated_memory
                    print(f"   GPU {gpu_id} free memory: {free_memory / 1024**3:.1f}GB")
                    
                    if free_memory < 2 * 1024**3:
                        print("Not enough GPU memory, loading on CPU...")
                        self.device = "cpu"
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float32,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            local_files_only=is_local
                        )
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            torch_dtype=self.torch_dtype,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            local_files_only=is_local
                        )
                        print("Weights loaded, moving to device...")
                        try:
                            self.model = self.model.to(self.device)
                            print(f"Model is on {self.device}")
                        except torch.cuda.OutOfMemoryError:
                            print("GPU out of memory, keeping the model on CPU")
                            self.device = "cpu"
            else:
                print("No GPU detected, loading on CPU...")
                self.device = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=is_local
                )
        
        print(f"Model is on {self.device}")

        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.output_dim = config.get("output_dim", 256)
        
        self.projection = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.output_dim)
        )
        
        self.projection = self.projection.to(self.device)
        
    def encode_text(self, text_inputs, return_thinking_process=False):
        """
        
        Args:
            
        Returns:
        """
        if isinstance(text_inputs, str):
            inputs = self.tokenizer(
                text_inputs,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True
            )
        elif isinstance(text_inputs, list):
            inputs = self.tokenizer(
                text_inputs,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True
            )
        else:
            inputs = text_inputs
            
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
        
        if next(self.model.parameters()).device != self.device:
            self.model = self.model.to(self.device)
        
        if next(self.projection.parameters()).device != self.device:
            self.projection = self.projection.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            if hasattr(self.model.config, 'cls_token_id'):
                pooled_output = last_hidden_states[:, 0, :]  # [batch_size, hidden_size]
            else:
                attention_mask = inputs.get('attention_mask', None)
                if attention_mask is not None:
                    pooled_output = (last_hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                else:
                    pooled_output = last_hidden_states.mean(dim=1)
            
            pooled_output = pooled_output.float()
            encoded_features = self.projection(pooled_output)
            
            if return_thinking_process:
                thinking_process = self._extract_thinking_process(outputs.logits, inputs)
                return encoded_features, thinking_process
            else:
                return encoded_features
    
    def _extract_thinking_process(self, logits, inputs):
        """
        """
        return logits
    
    def generate_response(self, prompt, max_new_tokens=512, temperature=0.7):
        """
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def forward(self, text_inputs):
        """
        """
        return self.encode_text(text_inputs)
