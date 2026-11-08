import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..base_models import TextEncoder

class QwenEncoder(TextEncoder):
    """
    
    用于处理非结构化知识和问题，支持思维链推理
    """
    def __init__(self, config):
        super(QwenEncoder, self).__init__()
        self.model_name = config.get("model_name", "Qwen/Qwen2-0.5B")
        self.max_length = config.get("max_length", 512)
        self.device = config.get("device", "cuda:0")
        
        # dtype：默认 bfloat16 更稳（尤其是 Qwen2.x/3.x），可在 config 里覆盖
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
        
        # 判断是本地路径还是 HuggingFace repo
        import os
        is_local = os.path.exists(self.model_name)
        
        # 加载Qwen模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=is_local
        )
        

        # 智能显存管理 - 根据可用显存选择加载策略
        print(f" 加载Qwen模型: {self.model_name}")
        print(f"   目标设备: {self.device}")
        
        # 如果指定了CPU，直接使用CPU加载
        if self.device == "cpu":
            print(" 使用CPU模式加载...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=is_local
            )
            print(" CPU模式加载成功")
        else:
            # 检查可用显存
            if torch.cuda.is_available():
                gpu_id = int(self.device.split(':')[1]) if ':' in self.device else 0
                # 检查GPU ID是否有效
                if gpu_id >= torch.cuda.device_count():
                    print(f" GPU {gpu_id} 不存在，使用CPU模式")
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
                    print(f"   GPU {gpu_id} 可用显存: {free_memory / 1024**3:.1f}GB")
                    
                    # 如果显存不足，使用CPU加载
                    if free_memory < 2 * 1024**3:  # 小于2GB
                        print(" 显存不足，使用CPU模式...")
                        self.device = "cpu"
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float32,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            local_files_only=is_local
                        )
                    else:
                        # 显存充足，正常加载
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            torch_dtype=self.torch_dtype,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            local_files_only=is_local
                        )
                        print(" 模型加载成功，正在移动到设备...")
                        # 尝试移动到GPU，如果失败则保持在CPU
                        try:
                            self.model = self.model.to(self.device)
                            print(f" 模型已移动到 {self.device}")
                        except torch.cuda.OutOfMemoryError:
                            print(" GPU显存不足，模型保持在CPU")
                            self.device = "cpu"
            else:
                # 没有GPU，使用CPU
                print(" 未检测到GPU，使用CPU模式...")
                self.device = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=is_local
                )
        
        print(f" 模型已移动到 {self.device}")

        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.output_dim = config.get("output_dim", 256)  # 小模型的隐藏维度
        
        # 投影层，将Qwen输出映射到期望维度
        self.projection = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.output_dim)
        )
        
        # 确保投影层在正确的设备上
        self.projection = self.projection.to(self.device)
        
    def encode_text(self, text_inputs, return_thinking_process=False):
        """
        编码文本输入，支持思维链推理
        
        Args:
            text_inputs: 文本输入，可以是字符串或tokenized输入
            return_thinking_process: 是否返回思维过程
            
        Returns:
            encoded_features: 编码后的特征向量
            thinking_process: 思维过程（可选）
        """
        if isinstance(text_inputs, str):
            # 如果是字符串，进行tokenization
            inputs = self.tokenizer(
                text_inputs,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True
            )
        elif isinstance(text_inputs, list):
            # 如果是字符串列表，进行tokenization
            inputs = self.tokenizer(
                text_inputs,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True
            )
        else:
            inputs = text_inputs
            
        # 移动到设备
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # 如果不是字典，尝试转换
            inputs = inputs.to(self.device)
        
        # 确保模型和投影层都在正确的设备上
        if next(self.model.parameters()).device != self.device:
            self.model = self.model.to(self.device)
        
        if next(self.projection.parameters()).device != self.device:
            self.projection = self.projection.to(self.device)
        
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # 使用最后一个隐藏状态
            last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            # 使用[CLS] token或平均池化
            if hasattr(self.model.config, 'cls_token_id'):
                # 如果有CLS token
                pooled_output = last_hidden_states[:, 0, :]  # [batch_size, hidden_size]
            else:
                # 使用平均池化
                attention_mask = inputs.get('attention_mask', None)
                if attention_mask is not None:
                    # 根据attention mask进行平均池化
                    pooled_output = (last_hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                else:
                    pooled_output = last_hidden_states.mean(dim=1)
            
            # 投影到目标维度
            # 确保数据类型匹配：将float16转换为float32
            pooled_output = pooled_output.float()
            encoded_features = self.projection(pooled_output)
            
            if return_thinking_process:
                # 提取思维过程（这里简化处理，实际可能需要更复杂的解析）
                thinking_process = self._extract_thinking_process(outputs.logits, inputs)
                return encoded_features, thinking_process
            else:
                return encoded_features
    
    def _extract_thinking_process(self, logits, inputs):
        """
        从模型输出中提取思维过程
        这是一个简化的实现，实际可能需要更复杂的解析逻辑
        """
        # 这里可以根据Qwen的思维链格式进行解析
        # 暂时返回原始logits作为思维过程
        return logits
    
    def generate_response(self, prompt, max_new_tokens=512, temperature=0.7):
        """
        生成响应，支持思维链推理
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
            
        # 解码生成的文本
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def forward(self, text_inputs):
        """
        前向传播，兼容现有接口
        """
        return self.encode_text(text_inputs)
