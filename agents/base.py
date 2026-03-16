import yaml, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from agents.logger import get_logger

class BaseAgent:
    def __init__(self, agent_name):
        try:
            with open("config.yaml") as _f:
                self.config = yaml.safe_load(_f) or {}
        except FileNotFoundError:
            raise FileNotFoundError("config.yaml not found — run from project root directory")
        except yaml.YAMLError as e:
            raise ValueError(f"config.yaml has invalid YAML syntax: {e}")
        self.agent_name = agent_name
        self.logger = get_logger(self.__class__.__name__)
        if agent_name not in self.config.get('models', {}):
            raise KeyError(f"Model key '{agent_name}' not found in config.yaml 'models' section")
        model_cfg = self.config['models'][agent_name]
        # Support both string (single model) and list (multi-model ensemble)
        if isinstance(model_cfg, list):
            self.model_paths = model_cfg
            self.model_path = model_cfg[0]
        else:
            self.model_paths = [model_cfg]
            self.model_path = model_cfg
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.quantization = self.config.get('quantization', 'none')
        self.model = None
        self.tokenizer = None
        self._loaded_path = None
        self._has_chat_template = None

    def load_model(self, model_path=None):
        path = model_path or self.model_path
        if self._loaded_path == path and self.model is not None:
            return
        if self.model is not None:
            self.free_vram()
        self.logger.info("Loading %s from %s ...", self.agent_name, path)

        quant_config = None
        if self.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True
        )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                quantization_config=quant_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        except AttributeError as e:
            if "max_length" in str(e):
                self.logger.warning("GLM-4 config issue (max_length), applying fix...")
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(path, trust_remote_code=True)
                config.max_length = config.seq_length
                self.model = AutoModelForCausalLM.from_pretrained(
                    path,
                    config=config,
                    quantization_config=quant_config,
                    device_map=self.device,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                raise

        # Detect chat template support
        self._has_chat_template = (
            hasattr(self.tokenizer, 'chat_template')
            and self.tokenizer.chat_template is not None
        )
        if self._has_chat_template:
            self.logger.info("%s: chat template detected, using apply_chat_template", self.agent_name)
        else:
            self.logger.info("%s: no chat template, using raw prompt", self.agent_name)

        self._loaded_path = path
        self.logger.info("%s loaded on %s", self.agent_name, self.device)

    def free_vram(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded_path = None
        self._has_chat_template = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    def _tokenize_prompt(self, prompt):
        """Tokenize a prompt, using chat template if available.

        If the tokenizer has a chat template (Qwen, LLaMA-Instruct, Mistral, etc.),
        wraps the prompt as a user message and applies the template. Otherwise,
        tokenizes the raw prompt string (works for base/completion models).

        Returns: dict with input_ids tensor on the correct device.
        """
        if self._has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return inputs

    def _decode_response(self, outputs, input_length):
        """Decode model output, stripping the input tokens.

        Uses token-level slicing (not string slicing) so it works correctly
        regardless of chat template special tokens.
        """
        new_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate(self, prompt, max_new_tokens=512, temperature=0.7):
        inputs = self._tokenize_prompt(prompt)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self._decode_response(outputs, input_length)
