"""
Manual Vertex AI service implementation using direct HTTP requests.
Replaces LangChain VertexAI components for better control and scalability.
"""

import os
import json
import requests
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VertexAILLM:
    """Manual Vertex AI LLM implementation using direct requests."""

    def __init__(self, endpoint_url: str, access_token: str, model_name: str = "sealion"):
        self.endpoint_url = endpoint_url
        self.access_token = access_token
        self.model_name = model_name

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Invoke the LLM with messages."""
        # Convert messages to the format expected by SEALion
        if isinstance(messages, list) and len(messages) > 0:
            role = messages[0].get("role", "user")
            content = messages[0].get("content", "")
        else:
            role = "user"
            content = str(messages)

        max_tokens = kwargs.get("max_tokens", 1024)
        temperature = kwargs.get("temperature", 0.6)
        top_p = kwargs.get("top_p", 1.0)
        top_k = kwargs.get("top_k", -1)

        request_data = {
            "instances": [
                {
                    "@requestFormat": "chatCompletions",
                    "messages": [{"role": role, "content": content}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=request_data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            # Extract the response content
            if "predictions" in result and len(result["predictions"]) > 0:
                return result["predictions"][0].get("content", "")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            raise Exception(f"LLM request failed: {e}")


class VertexAIEmbeddingsService:
    """Manual Vertex AI Embeddings implementation using direct requests."""

    def __init__(self, endpoint_url: str, access_token: str):
        self.endpoint_url = endpoint_url
        self.access_token = access_token

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        request_data = {
            "instances": [
                {"input": text}
            ]
        }
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        logger.info(f"Making request to: {self.endpoint_url}")
        logger.info(f"Request data: {request_data}")
        
        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=request_data,
                timeout=30  # Reduced timeout
            )
            logger.info(f"Response status: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            logger.info(f"Response: {result}")
            return result["predictions"][0]
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding request failed: {e}")
            raise Exception(f"Embedding request failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        request_data = {
            "instances": [{"input": text} for text in texts]
        }
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=request_data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["predictions"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Embeddings request failed: {e}")
            raise Exception(f"Embeddings request failed: {e}")


# Factory functions for dependency injection
def create_vertex_ai_llm():
    """Create Vertex AI LLM instance."""
    access_token = os.getenv("GCLOUD_TOKEN")
    endpoint_url = os.getenv("SEALION_MODEL_ENDPOINT")
    if not access_token or not endpoint_url:
        logger.warning("GCLOUD_TOKEN or SEALION_MODEL_ENDPOINT not set")
        return None
    return VertexAILLM(endpoint_url, access_token)


def create_vertex_ai_embeddings():
    """Create Vertex AI Embeddings instance."""
    access_token = os.getenv("GCLOUD_TOKEN")
    endpoint_url = os.getenv("EMBEDDING_MODEL_ENDPOINT")
    if not access_token or not endpoint_url:
        logger.warning("GCLOUD_TOKEN or EMBEDDING_MODEL_ENDPOINT not set")
        return None
    return VertexAIEmbeddingsService(endpoint_url, access_token)