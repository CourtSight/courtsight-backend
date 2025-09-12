"""
Centralized LLM Service for Supreme Court RAG System.
Provides unified interface for all LLM operations with centralized configuration management.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache
from contextlib import asynccontextmanager

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableSequence

from ..core.config import Settings, settings
from ..core.exceptions import LLMError

logger = logging.getLogger(__name__)

# Import Google Generative AI safety settings
try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    logger.warning("Google Generative AI not available, safety settings will be ignored")


class LLMCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for LLM operations."""

    def __init__(self):
        self.tokens_used = 0
        self.total_cost = 0.0

    def on_llm_start(self, serialized, prompts, **kwargs):
        logger.info(f"LLM call started with {len(prompts)} prompts")

    def on_llm_end(self, response, **kwargs):
        logger.info("LLM call completed")

    def on_llm_error(self, error, **kwargs):
        logger.error(f"LLM call failed: {str(error)}")


class LLMService:
    """
    Centralized LLM service with full configuration management.
    Handles all LLM operations including chat, embeddings, and custom chains.
    """

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or settings
        self._llm_instance: Optional[ChatGoogleGenerativeAI] = None
        self._embeddings_instance: Optional[GoogleGenerativeAIEmbeddings] = None
        self._callback_handler = LLMCallbackHandler()

        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize LLM and embeddings components."""
        try:
            self._llm_instance = self._create_llm_instance()
            self._embeddings_instance = self._create_embeddings_instance()
            logger.info("LLM service components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service components: {str(e)}")
            raise LLMError(f"LLM service initialization failed: {str(e)}")

    def _create_llm_instance(self) -> ChatGoogleGenerativeAI:
        """Create configured LLM instance with safety settings and performance configurations."""
        try:
            # Build safety settings if Google Generative AI is available
            safety_settings = None
            # if GOOGLE_GENAI_AVAILABLE:
            #     safety_settings = {
            #         HarmCategory.HARM_CATEGORY_HATE_SPEECH: self._get_harm_threshold(
            #             self.config.LLM_SAFETY_HATE_SPEECH
            #         ),
            #         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: self._get_harm_threshold(
            #             self.config.LLM_SAFETY_DANGEROUS_CONTENT
            #         ),
            #         HarmCategory.HARM_CATEGORY_HARASSMENT: self._get_harm_threshold(
            #             self.config.LLM_SAFETY_HARASSMENT
            #         ),
            #         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: self._get_harm_threshold(
            #             self.config.LLM_SAFETY_SEXUALLY_EXPLICIT
            #         ),
            #     }

            # Create LLM instance with all configurations
            llm_kwargs = {
                "model": self.config.LLM_MODEL_NAME,
                "temperature": self.config.LLM_TEMPERATURE,
                # "max_tokens": self.config.LLM_MAX_TOKENS,
                "google_api_key": self.config.GOOGLE_API_KEY.get_secret_value(),
                "verbose": self.config.LOG_LEVEL == "DEBUG",
                "callbacks": [self._callback_handler],
                # "candidate_count": self.config.LLM_CANDIDATE_COUNT,
                # "stop": self.config.LLM_STOP_SEQUENCES,
            }

            # # Add safety settings if available
            # if safety_settings:
            #     llm_kwargs["safety_settings"] = safety_settings

            # # Add performance settings
            # if hasattr(self.config, 'LLM_REQUEST_PARALLELISM'):
            #     llm_kwargs["request_parallelism"] = self.config.LLM_REQUEST_PARALLELISM

            # if hasattr(self.config, 'LLM_MAX_RETRIES'):
            #     llm_kwargs["max_retries"] = self.config.LLM_MAX_RETRIES

            return ChatGoogleGenerativeAI(**llm_kwargs)

        except Exception as e:
            logger.error(f"Failed to create LLM instance: {str(e)}")
            raise

    def _get_harm_threshold(self, threshold_str: str) -> "HarmBlockThreshold":
        """Convert string threshold to HarmBlockThreshold enum."""
        if not GOOGLE_GENAI_AVAILABLE:
            return None

        threshold_map = {
            "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
            "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        return threshold_map.get(threshold_str, HarmBlockThreshold.BLOCK_NONE)

    def _create_embeddings_instance(self) -> GoogleGenerativeAIEmbeddings:
        """Create configured embeddings instance."""
        try:
            return GoogleGenerativeAIEmbeddings(
                model=self.config.EMBEDDING_MODEL_NAME,
                google_api_key=self.config.GOOGLE_API_KEY.get_secret_value()
            )
        except Exception as e:
            logger.error(f"Failed to create embeddings instance: {str(e)}")
            raise

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Get the configured LLM instance."""
        if self._llm_instance is None:
            self._llm_instance = self._create_llm_instance()
        return self._llm_instance

    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Get the configured embeddings instance."""
        if self._embeddings_instance is None:
            self._embeddings_instance = self._create_embeddings_instance()
        return self._embeddings_instance

    def create_chat_chain(
        self,
        system_prompt: str,
        user_prompt_template: Optional[str] = None,
        output_parser: Optional[Any] = None
    ) -> RunnableSequence:
        """
        Create a chat chain with system prompt.

        Args:
            system_prompt: System message for the LLM
            user_prompt_template: Optional user prompt template
            output_parser: Optional output parser

        Returns:
            Configured RunnableSequence
        """
        try:
            if user_prompt_template:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("user", user_prompt_template)
                ])
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("user", "{input}")
                ])

            chain = prompt | self.llm

            if output_parser:
                chain = chain | output_parser

            return chain

        except Exception as e:
            logger.error(f"Failed to create chat chain: {str(e)}")
            raise LLMError(f"Chat chain creation failed: {str(e)}")

    def create_structured_chain(
        self,
        system_prompt: str,
        response_format: Any,
        user_prompt_template: Optional[str] = None
    ) -> RunnableSequence:
        """
        Create a structured output chain.

        Args:
            system_prompt: System message for the LLM
            response_format: Pydantic model for structured output
            user_prompt_template: Optional user prompt template

        Returns:
            Configured RunnableSequence with structured output
        """
        try:
            parser = JsonOutputParser(pydantic_object=response_format)

            format_instructions = parser.get_format_instructions()

            full_system_prompt = f"{system_prompt}\n\n{format_instructions}"

            if user_prompt_template:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", full_system_prompt),
                    ("user", user_prompt_template)
                ])
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", full_system_prompt),
                    ("user", "{input}")
                ])

            return prompt | self.llm | parser

        except Exception as e:
            logger.error(f"Failed to create structured chain: {str(e)}")
            raise LLMError(f"Structured chain creation failed: {str(e)}")

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional LLM parameters

        Returns:
            Generated text
        """
        try:
            if system_prompt:
                messages = [
                    ("system", system_prompt),
                    ("user", prompt)
                ]
                response = await self.llm.ainvoke(messages, **kwargs)
            else:
                response = await self.llm.ainvoke(prompt, **kwargs)

            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise LLMError(f"Text generation failed: {str(e)}")

    async def generate_structured_output(
        self,
        prompt: str,
        response_format: Any,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Generate structured output using the LLM.

        Args:
            prompt: User prompt
            response_format: Pydantic model for response format
            system_prompt: Optional system prompt
            **kwargs: Additional LLM parameters

        Returns:
            Structured response object
        """
        try:
            chain = self.create_structured_chain(
                system_prompt or "You are a helpful assistant.",
                response_format,
                "{input}"
            )

            result = await chain.ainvoke({"input": prompt}, **kwargs)
            return result

        except Exception as e:
            logger.error(f"Structured output generation failed: {str(e)}")
            raise LLMError(f"Structured output generation failed: {str(e)}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Text embedding failed: {str(e)}")
            raise LLMError(f"Text embedding failed: {str(e)}")

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            raise LLMError(f"Query embedding failed: {str(e)}")

    def get_service_health(self) -> Dict[str, Any]:
        """
        Get LLM service health status.

        Returns:
            Health status dictionary
        """
        try:
            health_info = {
                "service": "llm_service",
                "status": "healthy",
                "llm_model": self.config.LLM_MODEL_NAME,
                "embedding_model": self.config.EMBEDDING_MODEL_NAME,
                "temperature": self.config.LLM_TEMPERATURE,
                "max_tokens": self.config.LLM_MAX_TOKENS,
                "candidate_count": getattr(self.config, 'LLM_CANDIDATE_COUNT', 'N/A'),
                "request_parallelism": getattr(self.config, 'LLM_REQUEST_PARALLELISM', 'N/A'),
                "max_retries": getattr(self.config, 'LLM_MAX_RETRIES', 'N/A'),
                "safety_settings": {
                    "hate_speech": getattr(self.config, 'LLM_SAFETY_HATE_SPEECH', 'N/A'),
                    "dangerous_content": getattr(self.config, 'LLM_SAFETY_DANGEROUS_CONTENT', 'N/A'),
                    "harassment": getattr(self.config, 'LLM_SAFETY_HARASSMENT', 'N/A'),
                    "sexually_explicit": getattr(self.config, 'LLM_SAFETY_SEXUALLY_EXPLICIT', 'N/A'),
                },
                "vertex_ai_available": GOOGLE_GENAI_AVAILABLE,
                "components_initialized": {
                    "llm": self._llm_instance is not None,
                    "embeddings": self._embeddings_instance is not None
                }
            }

            # Test basic functionality
            try:
                # Simple embedding test
                test_embedding = self.embed_query("test")
                health_info["embedding_test"] = len(test_embedding) > 0
            except Exception as e:
                health_info["embedding_test"] = False
                health_info["embedding_error"] = str(e)

            return health_info

        except Exception as e:
            return {
                "service": "llm_service",
                "status": "unhealthy",
                "error": str(e)
            }

    def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """
        Update service configuration dynamically.

        Args:
            new_config: Dictionary of configuration updates
        """
        try:
            # Update config object
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            # Reinitialize components with new config
            self._initialize_components()

            logger.info(f"LLM service configuration updated: {new_config}")

        except Exception as e:
            logger.error(f"Configuration update failed: {str(e)}")
            raise LLMError(f"Configuration update failed: {str(e)}")


# Global service instance
_llm_service_instance: Optional[LLMService] = None


@lru_cache()
def get_llm_service() -> LLMService:
    """Get singleton LLM service instance."""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance


def create_llm_service(config: Optional[Settings] = None) -> LLMService:
    """
    Factory function to create LLM service instance.

    Args:
        config: Optional custom configuration

    Returns:
        Configured LLMService instance
    """
    return LLMService(config)


@asynccontextmanager
async def llm_service_lifecycle():
    """Context manager for LLM service lifecycle management."""
    service = get_llm_service()
    try:
        yield service
    finally:
        # Cleanup if needed
        logger.info("LLM service lifecycle ended")