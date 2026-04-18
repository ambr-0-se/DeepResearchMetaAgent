import os
from openai import OpenAI, AsyncOpenAI
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv(verbose=True)

from langchain_openai import ChatOpenAI

from src.logger import logger
from src.models.litellm import LiteLLMModel
from src.models.openaillm import OpenAIServerModel
from src.models.hfllm import InferenceClientModel
from src.models.restful import (RestfulModel,
                                RestfulTranscribeModel,
                                RestfulImagenModel,
                                RestfulVeoPridictModel,
                                RestfulVeoFetchModel,
                                RestfulResponseModel)
from src.models.failover import FailoverModel
from src.utils import Singleton
from src.proxy.local_proxy import HTTP_CLIENT, ASYNC_HTTP_CLIENT

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
PLACEHOLDER = "PLACEHOLDER"


class ModelManager(metaclass=Singleton):
    def __init__(self):
        self.registed_models: Dict[str, Any] = {}
        
    def init_models(self, use_local_proxy: bool = False):
        self._register_openai_models(use_local_proxy=use_local_proxy)
        self._register_anthropic_models(use_local_proxy=use_local_proxy)
        self._register_google_models(use_local_proxy=use_local_proxy)
        self._register_qwen_models(use_local_proxy=use_local_proxy)
        self._register_langchain_models(use_local_proxy=use_local_proxy)
        self._register_vllm_models(use_local_proxy=use_local_proxy)
        self._register_deepseek_models(use_local_proxy=use_local_proxy)
        self._register_dashscope_models()
        self._register_mistral_models()
        self._register_moonshot_models()
        self._register_minimax_models()
        self._register_openrouter_models()
        # Failover wrappers must register LAST — they read already-registered
        # primary + backup models and wrap them. Order: dashscope + openrouter
        # both run before this line.
        self._register_qwen_failover_models()

    def _check_local_api_key(self, local_api_key_name: str, remote_api_key_name: str) -> str:
        api_key = os.getenv(local_api_key_name, PLACEHOLDER)
        if api_key == PLACEHOLDER:
            logger.warning(f"Local API key {local_api_key_name} is not set, using remote API key {remote_api_key_name}")
            api_key = os.getenv(remote_api_key_name, PLACEHOLDER)
        return api_key
    
    def _check_local_api_base(self, local_api_base_name: str, remote_api_base_name: str) -> str:
        api_base = os.getenv(local_api_base_name, PLACEHOLDER)
        if api_base == PLACEHOLDER:
            logger.warning(f"Local API base {local_api_base_name} is not set, using remote API base {remote_api_base_name}")
            api_base = os.getenv(remote_api_base_name, PLACEHOLDER)
        return api_base

    @staticmethod
    def _api_key_configured(api_key: str | None) -> bool:
        """True when an API key is present (not placeholder / not whitespace-only)."""
        if api_key is None or api_key == PLACEHOLDER:
            return False
        if isinstance(api_key, str) and not api_key.strip():
            return False
        return True

    def _register_openai_models(self, use_local_proxy: bool = False):
        # gpt-4o, gpt-4.1, o1, o3, gpt-4o-search-preview
        if use_local_proxy:
            logger.info("Using local proxy for OpenAI models")
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY",
                                                remote_api_key_name="OPENAI_API_KEY")
            
            # gpt-4o
            model_name = "gpt-4o"
            model_id = "openai/gpt-4o"
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
                http_client=ASYNC_HTTP_CLIENT,
            )
            model = LiteLLMModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model
            
            # gpt-4.1
            model_name = "gpt-4.1"
            model_id = "openai/gpt-4.1"
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
                http_client=ASYNC_HTTP_CLIENT,
            )
            model = LiteLLMModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model
            
            # o1
            model_name = "o1"
            model_id = "openai/o1"
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
                http_client=ASYNC_HTTP_CLIENT,
            )
            model = LiteLLMModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model
            
            # o3
            model_name = "o3"
            model_id = "openai/o3"

            model = RestfulModel(
                api_base=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
                api_type="chat/completions",
                api_key=api_key,
                model_id=model_id,
                http_client=HTTP_CLIENT,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model
            
            # gpt-4o-search-preview
            model_name = "gpt-4o-search-preview"
            model_id = "gpt-4o-search-preview"
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_US_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE"),
                http_client=ASYNC_HTTP_CLIENT,
            )
            model = LiteLLMModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

            # wisper
            model_name = "whisper"
            model_id = "whisper"
            model = RestfulTranscribeModel(
                api_base=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_BJ_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
                api_key=api_key,
                api_type="whisper",
                model_id=model_id,
                http_client=HTTP_CLIENT,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

            # deep research
            model_name = "o3-deep-research"
            model_id = "o3-deep-research"

            model = RestfulResponseModel(
                api_base=self._check_local_api_base(local_api_base_name="SKYWORK_SHUBIAOBIAO_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
                api_key=api_key,
                api_type="responses",
                model_id=model_id,
                http_client=HTTP_CLIENT,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model
            
            # gpt-5
            model_name = "gpt-5"
            model_id = "openai/gpt-5"
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_AZURE_US_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE"),
                http_client=ASYNC_HTTP_CLIENT,
            )
            model = LiteLLMModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model
            
        else:
            logger.info("Using remote API for OpenAI models")
            api_key = self._check_local_api_key(local_api_key_name="OPENAI_API_KEY", 
                                                remote_api_key_name="OPENAI_API_KEY")
            if not self._api_key_configured(api_key):
                logger.warning(
                    "OPENAI_API_KEY is not set or empty, skipping OpenAI models "
                    "(GAIA matrix on Mistral/Kimi/Qwen is unaffected)"
                )
                return
            api_base = self._check_local_api_base(local_api_base_name="OPENAI_API_BASE", 
                                                    remote_api_base_name="OPENAI_API_BASE")
            
            models = [
                {
                    "model_name": "gpt-4o",
                    "model_id": "gpt-4o",
                },
                {
                    "model_name": "gpt-4.1",
                    "model_id": "gpt-4.1",
                },
                {
                    "model_name": "o1",
                    "model_id": "o1",
                },
                {
                    "model_name": "o3",
                    "model_id": "o3",
                },
                {
                    "model_name": "gpt-4o-search-preview",
                    "model_id": "gpt-4o-search-preview",
                },
            ]
            
            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]
                model = LiteLLMModel(
                    model_id=model_id,
                    api_key=api_key,
                    api_base=api_base,
                    custom_role_conversions=custom_role_conversions,
                )
                self.registed_models[model_name] = model
    
            
    def _register_anthropic_models(self, use_local_proxy: bool = False):
        # claude37-sonnet, claude37-sonnet-thinking
        if use_local_proxy:
            logger.info("Using local proxy for Anthropic models")
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY", 
                                                remote_api_key_name="ANTHROPIC_API_KEY")
            
            # claude37-sonnet
            model_name = "claude37-sonnet"
            model_id = "claude37-sonnet"
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_US_API_BASE",
                                                    remote_api_base_name="ANTHROPIC_API_BASE"),
                http_client=ASYNC_HTTP_CLIENT,
            )
            model = OpenAIServerModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model
            
            # claude37-sonnet-thinking
            model_name = "claude-3.7-sonnet-thinking"
            model_id = "claude-3.7-sonnet-thinking"
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_US_API_BASE",
                                                    remote_api_base_name="ANTHROPIC_API_BASE"),
                http_client=ASYNC_HTTP_CLIENT,
            )
            model = OpenAIServerModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

            # claude-4-sonnet
            model_name = "claude-4-sonnet"
            model_id = "claude-4-sonnet"
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_US_API_BASE",
                                                    remote_api_base_name="ANTHROPIC_API_BASE"),
                http_client=ASYNC_HTTP_CLIENT,
            )
            model = OpenAIServerModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

        else:
            logger.info("Using remote API for Anthropic models")
            api_key = self._check_local_api_key(local_api_key_name="ANTHROPIC_API_KEY", 
                                                remote_api_key_name="ANTHROPIC_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="ANTHROPIC_API_BASE", 
                                                    remote_api_base_name="ANTHROPIC_API_BASE")
            
            models = [
                {
                    "model_name": "claude37-sonnet",
                    "model_id": "claude-3-7-sonnet-20250219",
                },
                {
                    "model_name": "claude37-sonnet-thinking",
                    "model_id": "claude-3-7-sonnet-20250219",
                },
            ]
            
            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]
                model = LiteLLMModel(
                    model_id=model_id,
                    api_key=api_key,
                    api_base=api_base,
                    custom_role_conversions=custom_role_conversions,
                )
                self.registed_models[model_name] = model
            
    def _register_google_models(self, use_local_proxy: bool = False):
        if use_local_proxy:
            logger.info("Using local proxy for Google models")
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY", 
                                                remote_api_key_name="GOOGLE_API_KEY")
            
            # gemini-2.5-pro
            model_name = "gemini-2.5-pro"
            model_id = "gemini-2.5-pro-preview-06-05"
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._check_local_api_base(local_api_base_name="SKYWORK_OPENROUTER_BJ_API_BASE",
                                                    remote_api_base_name="GOOGLE_API_BASE"),
                http_client=ASYNC_HTTP_CLIENT,
            )
            model = OpenAIServerModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

            # imagen
            model_name = "imagen"
            model_id = "imagen-3.0-generate-001"
            model = RestfulImagenModel(
                api_base=self._check_local_api_base(local_api_base_name="SKYWORK_GOOGLE_API_BASE",
                                                    remote_api_base_name="GOOGLE_API_BASE"),
                api_key=api_key,
                api_type="imagen",
                model_id=model_id,
                http_client=HTTP_CLIENT,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

            # veo3
            model_name = "veo3-predict"
            model_id = "veo-3.0-generate-preview"
            model = RestfulVeoPridictModel(
                api_base=self._check_local_api_base(local_api_base_name="SKYWORK_GOOGLE_API_BASE",
                                                    remote_api_base_name="GOOGLE_API_BASE"),
                api_key=api_key,
                api_type="veo/predict",
                model_id=model_id,
                http_client=HTTP_CLIENT,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

            model_name = "veo3-fetch"
            model_id = "veo-3.0-generate-preview"
            model = RestfulVeoFetchModel(
                api_base=self._check_local_api_base(local_api_base_name="SKYWORK_GOOGLE_API_BASE",
                                                    remote_api_base_name="GOOGLE_API_BASE"),
                api_key=api_key,
                api_type="veo/fetch",
                model_id=model_id,
                http_client=HTTP_CLIENT,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

            
        else:
            logger.info("Using remote API for Google models")
            api_key = self._check_local_api_key(local_api_key_name="GOOGLE_API_KEY", 
                                                remote_api_key_name="GOOGLE_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="GOOGLE_API_BASE", 
                                                    remote_api_base_name="GOOGLE_API_BASE")
            
            models = [
                {
                    "model_name": "gemini-2.5-pro",
                    "model_id": "gemini-2.5-pro-preview-06-05",
                },
            ]
            
            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]
                model = LiteLLMModel(
                    model_id=model_id,
                    api_key=api_key,
                    # api_base=api_base,
                    custom_role_conversions=custom_role_conversions,
                )
                self.registed_models[model_name] = model
                
    def _register_qwen_models(self, use_local_proxy: bool = False):
        # qwen2.5-7b-instruct
        models = [
            {
                "model_name": "qwen2.5-7b-instruct",
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
            },
            {
                "model_name": "qwen2.5-14b-instruct",
                "model_id": "Qwen/Qwen2.5-14B-Instruct",
            },
            {
                "model_name": "qwen2.5-32b-instruct",
                "model_id": "Qwen/Qwen2.5-32B-Instruct",
            },
        ]
        for model in models:
            model_name = model["model_name"]
            model_id = model["model_id"]
            
            model = InferenceClientModel(
                model_id=model_id,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

    def _register_langchain_models(self, use_local_proxy: bool = False):
        # langchain models
        models = [
            {
                "model_name": "langchain-gpt-4o",
                "model_id": "gpt-4o",
            },
            {
                "model_name": "langchain-gpt-4.1",
                "model_id": "gpt-4.1",
            },
            {
                "model_name": "langchain-o3",
                "model_id": "o3",
            },
        ]

        if use_local_proxy:
            logger.info("Using local proxy for LangChain models")
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY",
                                                remote_api_key_name="OPENAI_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="SKYWORK_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE")

            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]

                model = ChatOpenAI(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base,
                    http_client=HTTP_CLIENT,
                    http_async_client=ASYNC_HTTP_CLIENT,
                )
                self.registed_models[model_name] = model

        else:
            logger.info("Using remote API for LangChain models")
            api_key = self._check_local_api_key(local_api_key_name="OPENAI_API_KEY",
                                                remote_api_key_name="OPENAI_API_KEY")
            if not self._api_key_configured(api_key):
                logger.warning(
                    "OPENAI_API_KEY is not set or empty, skipping LangChain OpenAI wrappers "
                    "(langchain-gpt-4o / langchain-gpt-4.1 / langchain-o3)"
                )
                return
            api_base = self._check_local_api_base(local_api_base_name="OPENAI_API_BASE",
                                                    remote_api_base_name="OPENAI_API_BASE")

            for model in models:
                model_name = model["model_name"]
                model_id = model["model_id"]

                model = ChatOpenAI(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base,
                )
                self.registed_models[model_name] = model
    def _register_vllm_models(self, use_local_proxy: bool = False):
        # qwen
        api_key = self._check_local_api_key(local_api_key_name="QWEN_API_KEY", 
                                                remote_api_key_name="QWEN_API_KEY")
        api_base = self._check_local_api_base(local_api_base_name="QWEN_API_BASE", 
                                                    remote_api_base_name="QWEN_API_BASE")
        models = [
            {
                "model_name": "Qwen",
                "model_id": "Qwen",
            }
        ]
        for model in models:
            model_name = model["model_name"]
            model_id = model["model_id"]
            
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
            )
            model = OpenAIServerModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

        # Add langchain wrapper for Qwen (required for browser-use tool)
        langchain_qwen = ChatOpenAI(
            model="Qwen",
            api_key=api_key,
            base_url=api_base,
        )
        self.registed_models["langchain-Qwen"] = langchain_qwen

        # Qwen-VL
        api_key_VL = self._check_local_api_key(local_api_key_name="QWEN_VL_API_KEY", 
                                                remote_api_key_name="QWEN_VL_API_KEY")
        api_base_VL = self._check_local_api_base(local_api_base_name="QWEN_VL_API_BASE", 
                                                    remote_api_base_name="QWEN_VL_API_BASE")
        models = [
            {
                "model_name": "Qwen-VL",
                "model_id": "Qwen-VL",
            }
        ]
        for model in models:
            model_name = model["model_name"]
            model_id = model["model_id"]

            client = AsyncOpenAI(
                api_key=api_key_VL,
                base_url=api_base_VL,
            )
            model = OpenAIServerModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

    def _register_deepseek_models(self, use_local_proxy: bool = False):
        # deepseek models
        if use_local_proxy:
            # deepseek-chat
            logger.info("Using local proxy for DeepSeek models")
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY",
                                                remote_api_key_name="SKYWORK_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="SKYWORK_DEEPSEEK_API_BASE",
                                                  remote_api_base_name="SKYWORK_API_BASE")

            model_name = "deepseek-chat"
            model_id = "deepseek-chat"
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
                http_client=ASYNC_HTTP_CLIENT,
            )
            model = OpenAIServerModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model

            # deepseek-reasoner
            api_key = self._check_local_api_key(local_api_key_name="SKYWORK_API_KEY",
                                                remote_api_key_name="SKYWORK_API_KEY")
            api_base = self._check_local_api_base(local_api_base_name="SKYWORK_DEEPSEEK_API_BASE",
                                                    remote_api_base_name="SKYWORK_API_BASE")

            model_name = "deepseek-reasoner"
            model_id = "deepseek-reasoner"
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
                http_client=ASYNC_HTTP_CLIENT,
            )
            model = OpenAIServerModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = model
        else:
            # Native DeepSeek API (https://api.deepseek.com).
            # OpenAI-SDK compatible. Supports deepseek-chat (V3.2) and
            # deepseek-reasoner (V3.2 with chain-of-thought reasoning_content).
            api_key = os.getenv("DEEPSEEK_API_KEY", PLACEHOLDER)
            if api_key == PLACEHOLDER:
                logger.warning(
                    "DEEPSEEK_API_KEY is not set, skipping native DeepSeek models"
                )
                return

            api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
            logger.info("Registering native DeepSeek models")

            models = [
                {"model_name": "deepseek-chat", "model_id": "deepseek-chat"},
                {"model_name": "deepseek-reasoner", "model_id": "deepseek-reasoner"},
            ]
            for m in models:
                model_name = m["model_name"]
                model_id = m["model_id"]
                client = AsyncOpenAI(api_key=api_key, base_url=api_base)
                registered_model = OpenAIServerModel(
                    model_id=model_id,
                    http_client=client,
                    custom_role_conversions=custom_role_conversions,
                )
                self.registed_models[model_name] = registered_model

                langchain_model = ChatOpenAI(
                    model=model_id,
                    api_key=api_key,
                    base_url=api_base,
                )
                self.registed_models[f"langchain-{model_name}"] = langchain_model

    def _register_dashscope_models(self):
        """Native Alibaba DashScope (Qwen) OpenAI-compatible endpoint.

        Covers Qwen3 Max / Qwen3.6 Plus / Qwen3-Coder-Plus etc. Uses the
        international endpoint by default; override via DASHSCOPE_API_BASE.
        Per-request `enable_thinking` must be threaded through config when
        reasoning mode is desired — not a constructor concern here.
        """
        api_key = os.getenv("DASHSCOPE_API_KEY", PLACEHOLDER)
        if api_key == PLACEHOLDER:
            logger.warning(
                "DASHSCOPE_API_KEY is not set, skipping DashScope (Qwen3) models"
            )
            return

        api_base = os.getenv(
            "DASHSCOPE_API_BASE",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        logger.info("Registering DashScope (Qwen3) models")

        # Each base model also registers a `-thinking` variant via extra_body to
        # switch DashScope reasoning mode on. The reasoning_content echo happens
        # automatically via MessageManager.needs_reasoning_echo() because the
        # thinking variant's model_id contains both "qwen3" and "thinking".
        #
        # Base (non-thinking) variants must EXPLICITLY send enable_thinking=False
        # because DashScope defaults some models (e.g. qwen3.6-plus) to thinking
        # mode server-side. Thinking mode rejects `tool_choice="required"` with
        # 400 InternalError.Algo.InvalidParameter, breaking the tool-calling
        # agent loop used by this project.
        models = [
            {"name": "qwen3-max", "id": "qwen3-max", "extra_body": {"enable_thinking": False}},
            {"name": "qwen3-max-thinking", "id": "qwen3-max", "extra_body": {"enable_thinking": True}},
            {"name": "qwen3.6-plus", "id": "qwen3.6-plus", "extra_body": {"enable_thinking": False}},
            {"name": "qwen3.6-plus-thinking", "id": "qwen3.6-plus", "extra_body": {"enable_thinking": True}},
            {"name": "qwen-plus", "id": "qwen-plus", "extra_body": {"enable_thinking": False}},
            {"name": "qwen3-coder-plus", "id": "qwen3-coder-plus", "extra_body": {"enable_thinking": False}},
        ]
        for m in models:
            model_name = m["name"]
            model_id = m["id"]
            client = AsyncOpenAI(api_key=api_key, base_url=api_base)
            registered_model = OpenAIServerModel(
                model_id=model_name if m["extra_body"] else model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
                extra_body=m["extra_body"],
            )
            # Preserve the actual model_id DashScope expects on the wire.
            # We only used model_name above so needs_reasoning_echo() sees "thinking".
            registered_model.model_id = model_id
            # But expose the friendly alias for echo-routing: stash on a side attr
            # that MessageManager can read if needed (it uses self.model_id by default,
            # so the approach above collides). Simpler: keep MessageManager's predicate
            # keyed on the alias by setting message_manager.model_id to the alias.
            registered_model.message_manager.model_id = model_name
            self.registed_models[model_name] = registered_model

            langchain_model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=api_base,
            )
            self.registed_models[f"langchain-{model_name}"] = langchain_model

    def _register_mistral_models(self):
        """Native Mistral La Plateforme (OpenAI-compatible).

        Covers Mistral Small 4 (`mistral-small-2603`). Tool results use role
        `tool`; parallel tool calls opt-in via `parallel_tool_calls` per request.
        """
        api_key = os.getenv("MISTRAL_API_KEY", PLACEHOLDER)
        if api_key == PLACEHOLDER:
            logger.warning("MISTRAL_API_KEY is not set, skipping Mistral models")
            return

        api_base = os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
        logger.info("Registering Mistral models")

        models = [
            {"model_name": "mistral-small", "model_id": "mistral-small-2603"},
            {"model_name": "mistral-small-latest", "model_id": "mistral-small-latest"},
        ]
        for m in models:
            model_name = m["model_name"]
            model_id = m["model_id"]
            client = AsyncOpenAI(api_key=api_key, base_url=api_base)
            registered_model = OpenAIServerModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = registered_model

            langchain_model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=api_base,
            )
            self.registed_models[f"langchain-{model_name}"] = langchain_model

    def _register_moonshot_models(self):
        """Native Moonshot (Kimi) OpenAI-compatible endpoint.

        Covers Kimi K2.5. Note: Moonshot fixes temperature (1.0 thinking /
        0.6 non-thinking) and top_p (0.95); callers must avoid overriding
        these sampling params. Thinking mode is on by default.
        """
        api_key = os.getenv("MOONSHOT_API_KEY", PLACEHOLDER)
        if api_key == PLACEHOLDER:
            logger.warning(
                "MOONSHOT_API_KEY is not set, skipping Moonshot (Kimi) models"
            )
            return

        api_base = os.getenv("MOONSHOT_API_BASE", "https://api.moonshot.ai/v1")
        logger.info("Registering Moonshot (Kimi) models")

        models = [
            {"model_name": "kimi-k2.5", "model_id": "kimi-k2.5", "extra_body": None},
            # Thinking-disabled variant — REQUIRED for C3 ReviewAgent and C4
            # SkillExtractor because Moonshot disallows response_format=json_object
            # while thinking is on (which is the default). Use this alias for any
            # config where the model must emit structured JSON.
            {
                "model_name": "kimi-k2.5-no-thinking",
                "model_id": "kimi-k2.5",
                "extra_body": {"thinking": {"type": "disabled"}},
            },
        ]
        for m in models:
            model_name = m["model_name"]
            model_id = m["model_id"]
            client = AsyncOpenAI(api_key=api_key, base_url=api_base)
            registered_model = OpenAIServerModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
                extra_body=m.get("extra_body"),
            )
            self.registed_models[model_name] = registered_model

            langchain_model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=api_base,
            )
            self.registed_models[f"langchain-{model_name}"] = langchain_model

    def _register_minimax_models(self):
        """Native MiniMax OpenAI-compatible endpoint.

        Covers MiniMax-M2.7 (and highspeed variant). IMPORTANT: M2.7 emits
        interleaved reasoning inside <think></think> that MUST be preserved
        across tool-calling turns. Temperature must be in (0, 1.0]; n=1 only.
        """
        api_key = os.getenv("MINIMAX_API_KEY", PLACEHOLDER)
        if api_key == PLACEHOLDER:
            logger.warning("MINIMAX_API_KEY is not set, skipping MiniMax models")
            return

        api_base = os.getenv("MINIMAX_API_BASE", "https://api.minimax.io/v1")
        logger.info("Registering MiniMax models")

        models = [
            {"model_name": "minimax-m2.7", "model_id": "MiniMax-M2.7"},
            {"model_name": "minimax-m2.7-highspeed", "model_id": "MiniMax-M2.7-highspeed"},
        ]
        for m in models:
            model_name = m["model_name"]
            model_id = m["model_id"]
            client = AsyncOpenAI(api_key=api_key, base_url=api_base)
            registered_model = OpenAIServerModel(
                model_id=model_id,
                http_client=client,
                custom_role_conversions=custom_role_conversions,
            )
            self.registed_models[model_name] = registered_model

            langchain_model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=api_base,
            )
            self.registed_models[f"langchain-{model_name}"] = langchain_model

    def _register_openrouter_models(self):
        api_key = os.getenv("OPENROUTER_API_KEY", PLACEHOLDER)
        if api_key == PLACEHOLDER:
            logger.warning("OPENROUTER_API_KEY is not set, skipping OpenRouter models")
            return

        api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        logger.info("Registering OpenRouter models")

        models = [
            # Free 117B MoE model with native tool use and configurable reasoning depth
            {
                "model_name": "gpt-oss-120b",
                "model_id": "openai/gpt-oss-120b:free",
            },
            # DeepSeek V3.2 — reasoner: must echo reasoning_content across tool turns
            {"model_name": "or-deepseek-v3.2", "model_id": "deepseek/deepseek-v3.2"},
            {"model_name": "or-deepseek-v3.2-exp", "model_id": "deepseek/deepseek-v3.2-exp"},
            # Mistral Small 4
            {"model_name": "or-mistral-small", "model_id": "mistralai/mistral-small-2603"},
            # Qwen3 family
            {"model_name": "or-qwen3-max", "model_id": "qwen/qwen3-max"},
            {"model_name": "or-qwen3.6-plus", "model_id": "qwen/qwen3.6-plus"},
            # Qwen3-Next 80B A3B Instruct — chosen Qwen slot in the eval matrix
            # after DashScope free tier exhausted and `or-qwen3.6-plus`'s
            # OpenRouter providers turned out not to support
            # `tool_choice="required"`. Live-verified 2026-04-18: tool_choice
            # required + multi-turn tool-result round-trip both succeed.
            {"model_name": "or-qwen3-next-80b-a3b-instruct", "model_id": "qwen/qwen3-next-80b-a3b-instruct"},
            {"model_name": "or-qwen3-coder-next", "model_id": "qwen/qwen3-coder-next"},
            # Moonshot Kimi K2.5 — fixed sampling params (temp/top_p locked)
            # extra_body: `thinking: disabled` satisfies Moonshot's constraint
            # that `tool_choice="required"` requires thinking off (else 400).
            # `provider.order=["Moonshot"]` pins routing to Moonshot so free-tier
            # OR cannot silently fall back to a sub-provider with diverging
            # thinking / sampling semantics.
            {
                "model_name": "or-kimi-k2.5",
                "model_id": "moonshotai/kimi-k2.5",
                "extra_body": {
                    "thinking": {"type": "disabled"},
                    "provider": {"order": ["Moonshot"]},
                },
            },
            # MiniMax M2.7 — preserve <think> blocks across tool turns
            {"model_name": "or-minimax-m2.7", "model_id": "minimax/minimax-m2.7"},
            # Google Gemma 4 31B Instruct (D4, 2026-04-18). Paid only — the
            # `:free` variant routes through Google AI Studio, which has the
            # least reliable tool-use support. Provider pin restricts routing
            # to DeepInfra + Together (both vLLM-backed, latest gemma4 parser)
            # so Novita (no `tools` support) cannot be selected. Reasoning
            # mode is disabled to prevent thinking-channel contamination of
            # tool output (vLLM issue #39043). Concurrency cap of 4 lives in
            # `scripts/run_eval_matrix.sh` cell_cmd (vLLM #39392 pad-bug under
            # parallel load). Live smoke probe 2026-04-18 confirmed
            # `tool_choice="required"` works with this provider pin
            # (finish_reason="tool_calls", no special-token leaks), so Gemma
            # is NOT in `MODELS_REJECTING_REQUIRED`.
            {
                "model_name": "or-gemma-4-31b-it",
                "model_id": "google/gemma-4-31b-it",
                "extra_body": {
                    "provider": {
                        "order": ["DeepInfra", "Together"],
                        "allow_fallbacks": False,
                    },
                    "reasoning": {"enabled": False},
                },
            },
        ]

        for model in models:
            model_name = model["model_name"]
            model_id = model["model_id"]
            extra_body = model.get("extra_body")

            client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
            )
            openai_kwargs: dict[str, Any] = {
                "model_id": model_id,
                "http_client": client,
                "custom_role_conversions": custom_role_conversions,
            }
            if extra_body is not None:
                openai_kwargs["extra_body"] = extra_body
            registered_model = OpenAIServerModel(**openai_kwargs)
            self.registed_models[model_name] = registered_model

            # LangChain wrapper required by auto_browser_use_tool.
            # Note: ChatOpenAI does not accept `extra_body` at construction;
            # provider-specific routing (thinking off, provider pin, etc.) is
            # not propagated here. If a browser-use agent starts exercising
            # those code paths under Kimi/Gemma/etc., plumb `model_kwargs`
            # through at that call site — out of scope for this change.
            langchain_model = ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=api_base,
            )
            self.registed_models[f"langchain-{model_name}"] = langchain_model

    def _register_qwen_failover_models(self):
        """Register `qwen3.6-plus-failover` if both DashScope and OpenRouter are
        configured. Routes calls to DashScope first (free tier) and switches to
        OpenRouter on quota-exhaustion errors. Switch is one-way per process.

        See `src/models/failover.py` for the detection heuristics.
        """
        primary = self.registed_models.get("qwen3.6-plus")
        backup = self.registed_models.get("or-qwen3.6-plus")
        if primary is None or backup is None:
            logger.info(
                "Skipping qwen3.6-plus-failover registration "
                "(primary=%s, backup=%s)",
                "set" if primary else "missing",
                "set" if backup else "missing",
            )
            return

        alias = "qwen3.6-plus-failover"
        self.registed_models[alias] = FailoverModel(
            primary=primary, backup=backup, alias=alias,
        )
        # LangChain wrapper for browser-use tool: cannot wrap FailoverModel
        # (it's not a ChatModel). Default to DashScope; on quota exhaustion
        # the browser-use sub-agent's LangChain calls will hard-fail and the
        # operator should switch the config to `or-qwen3.6-plus` for that role.
        # Explicitly register an alias so configs can name it.
        if "langchain-qwen3.6-plus" in self.registed_models:
            self.registed_models[f"langchain-{alias}"] = self.registed_models[
                "langchain-qwen3.6-plus"
            ]
        logger.info(
            "Registered %s (primary=qwen3.6-plus / DashScope, "
            "backup=or-qwen3.6-plus / OpenRouter)",
            alias,
        )