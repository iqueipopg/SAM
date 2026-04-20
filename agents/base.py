import json, os, yaml, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from agents.logger import get_logger

# Default path for per-agent JSON config (relative to project root)
_AGENTS_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "config", "agents.json")


class BaseAgent:
    """Base class for all LLM-powered pipeline agents.

    Input: agent_name (str) used to look up model path in config.yaml.
    Output: provides load_model(), generate(), free_vram() to subclasses.
    GPU: yes -- loads a quantized causal LM into VRAM.
    """

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

        # Load per-agent config from config/agents.json (if it exists)
        self.agent_params = self._load_agent_params(agent_name)

        # Determine backend: "local" (default) or "groq"
        groq_cfg = self.config.get("groq", {})
        self._backend = groq_cfg.get("provider", "local")
        self._groq_provider = None

        if self._backend == "groq":
            # Cloud backend — no local model needed
            from agents.groq_adapter import GroqProvider
            self._groq_provider = GroqProvider(self.config, agent_name=agent_name)
            self.logger.info(
                "%s: using Groq cloud backend (model=%s)",
                agent_name, groq_cfg.get("model", "llama-3.3-70b-versatile"),
            )
            # Set minimal attributes so subclasses don't break
            model_cfg = self.config.get('models', {}).get(agent_name, "groq-cloud")
            if isinstance(model_cfg, list):
                self.model_paths = model_cfg
                self.model_path = model_cfg[0]
            else:
                self.model_paths = [model_cfg]
                self.model_path = model_cfg
            self.device = "cloud"
            self.quantization = "none"
            gen_cfg = self.config.get('generation', {})
            self.do_sample = gen_cfg.get('do_sample', True)
            self.top_p = gen_cfg.get('top_p', 0.9)
            self.repetition_penalty = gen_cfg.get('repetition_penalty', 1.3)
            self.model = None
            self.tokenizer = None
            self._loaded_path = None
            self._has_chat_template = None
            return

        # Local backend — original behaviour
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
        gen_cfg = self.config.get('generation', {})
        self.do_sample = gen_cfg.get('do_sample', True)
        self.top_p = gen_cfg.get('top_p', 0.9)
        self.repetition_penalty = gen_cfg.get('repetition_penalty', 1.3)
        self.model = None
        self.tokenizer = None
        self._loaded_path = None
        self._has_chat_template = None

    @staticmethod
    def _load_agent_params(agent_name):
        """Load per-agent params from config/agents.json.

        Returns a dict with agent-specific overrides merged over defaults.
        If agents.json is missing, returns an empty dict (backward compat).
        """
        if not os.path.isfile(_AGENTS_JSON_PATH):
            return {}
        try:
            with open(_AGENTS_JSON_PATH) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
        defaults = data.get("defaults", {})
        agent_cfg = data.get("agents", {}).get(agent_name, {})
        merged = dict(defaults)
        merged.update(agent_cfg)
        return merged

    def get_agent_param(self, key, fallback=None):
        """Get a per-agent parameter from agents.json, with fallback.

        Lookup order:
        1. agents.json agent-specific value
        2. agents.json defaults
        3. Provided fallback value
        """
        return self.agent_params.get(key, fallback)

    def load_model(self, model_path=None):
        if self._backend == "groq":
            # Cloud backend — no local model to load
            return
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

        load_kwargs = dict(
            quantization_config=quant_config,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        for attempt in range(3):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
                break
            except AttributeError as e:
                if "max_length" in str(e):
                    self.logger.warning("GLM-4 config issue (max_length), applying fix...")
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
                    config.max_length = config.seq_length
                    self.model = AutoModelForCausalLM.from_pretrained(
                        path, config=config, **load_kwargs
                    )
                    break
                else:
                    raise
            except (BrokenPipeError, OSError) as e:
                self.logger.warning(
                    "load_model attempt %d/3 failed: %s. Clearing CUDA and retrying...",
                    attempt + 1, e,
                )
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                if attempt == 2:
                    raise RuntimeError(
                        f"Failed to load model after 3 attempts: {e}. "
                        "Try restarting Streamlit to reset CUDA context."
                    ) from e

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
        if self._backend == "groq":
            # Cloud backend — no VRAM to free
            return
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

    def _is_groq(self):
        """Check if using Groq cloud backend."""
        return self._backend == "groq"

    def generate(self, prompt, max_new_tokens=512, temperature=0.7):
        if self._is_groq():
            return self._groq_provider.generate(prompt, max_new_tokens, temperature)
        inputs = self._tokenize_prompt(prompt)
        input_length = inputs["input_ids"].shape[1]
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.do_sample,
            temperature=temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.repetition_penalty
        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)
        return self._decode_response(outputs, input_length)
