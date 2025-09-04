import os
from enum import Enum

from pydantic import SecretStr
from pydantic_settings import BaseSettings
from starlette.config import Config

current_file_dir = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(current_file_dir, "..", "..", ".env")
config = Config(env_path)


class AppSettings(BaseSettings):
    APP_NAME: str = config("APP_NAME", default="FastAPI app")
    APP_DESCRIPTION: str | None = config("APP_DESCRIPTION", default=None)
    APP_VERSION: str | None = config("APP_VERSION", default=None)
    LICENSE_NAME: str | None = config("LICENSE", default=None)
    CONTACT_NAME: str | None = config("CONTACT_NAME", default=None)
    CONTACT_EMAIL: str | None = config("CONTACT_EMAIL", default=None)


class CryptSettings(BaseSettings):
    SECRET_KEY: SecretStr = config("SECRET_KEY", cast=SecretStr)
    ALGORITHM: str = config("ALGORITHM", default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = config("ACCESS_TOKEN_EXPIRE_MINUTES", default=30)
    REFRESH_TOKEN_EXPIRE_DAYS: int = config("REFRESH_TOKEN_EXPIRE_DAYS", default=7)


class DatabaseSettings(BaseSettings):
    pass


class SQLiteSettings(DatabaseSettings):
    SQLITE_URI: str = config("SQLITE_URI", default="./sql_app.db")
    SQLITE_SYNC_PREFIX: str = config("SQLITE_SYNC_PREFIX", default="sqlite:///")
    SQLITE_ASYNC_PREFIX: str = config("SQLITE_ASYNC_PREFIX", default="sqlite+aiosqlite:///")


class MySQLSettings(DatabaseSettings):
    MYSQL_USER: str = config("MYSQL_USER", default="username")
    MYSQL_PASSWORD: str = config("MYSQL_PASSWORD", default="password")
    MYSQL_SERVER: str = config("MYSQL_SERVER", default="localhost")
    MYSQL_PORT: int = config("MYSQL_PORT", default=5432)
    MYSQL_DB: str = config("MYSQL_DB", default="dbname")
    MYSQL_URI: str = f"{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_SERVER}:{MYSQL_PORT}/{MYSQL_DB}"
    MYSQL_SYNC_PREFIX: str = config("MYSQL_SYNC_PREFIX", default="mysql://")
    MYSQL_ASYNC_PREFIX: str = config("MYSQL_ASYNC_PREFIX", default="mysql+aiomysql://")
    MYSQL_URL: str | None = config("MYSQL_URL", default=None)


class PostgresSettings(DatabaseSettings):
    POSTGRES_USER: str = config("POSTGRES_USER", default="postgres")
    POSTGRES_PASSWORD: str = config("POSTGRES_PASSWORD", default="postgres")
    POSTGRES_SERVER: str = config("POSTGRES_SERVER", default="localhost")
    POSTGRES_PORT: int = config("POSTGRES_PORT", default=5432)
    POSTGRES_DB: str = config("POSTGRES_DB", default="postgres")
    POSTGRES_SYNC_PREFIX: str = config("POSTGRES_SYNC_PREFIX", default="postgresql://")
    POSTGRES_ASYNC_PREFIX: str = config("POSTGRES_ASYNC_PREFIX", default="postgresql+asyncpg://")
    POSTGRES_URI: str = f"{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"
    POSTGRES_URL: str | None = config("POSTGRES_URL", default=None)


class FirstUserSettings(BaseSettings):
    ADMIN_NAME: str = config("ADMIN_NAME", default="admin")
    ADMIN_EMAIL: str = config("ADMIN_EMAIL", default="admin@admin.com")
    ADMIN_USERNAME: str = config("ADMIN_USERNAME", default="admin")
    ADMIN_PASSWORD: str = config("ADMIN_PASSWORD", default="!Ch4ng3Th1sP4ssW0rd!")


class TestSettings(BaseSettings): ...


class RedisCacheSettings(BaseSettings):
    REDIS_CACHE_HOST: str = config("REDIS_CACHE_HOST", default="localhost")
    REDIS_CACHE_PORT: int = config("REDIS_CACHE_PORT", default=6379)
    REDIS_CACHE_URL: str = f"redis://{REDIS_CACHE_HOST}:{REDIS_CACHE_PORT}"


class ClientSideCacheSettings(BaseSettings):
    CLIENT_CACHE_MAX_AGE: int = config("CLIENT_CACHE_MAX_AGE", default=60)


class RedisQueueSettings(BaseSettings):
    REDIS_QUEUE_HOST: str = config("REDIS_QUEUE_HOST", default="localhost")
    REDIS_QUEUE_PORT: int = config("REDIS_QUEUE_PORT", default=6379)


class RedisRateLimiterSettings(BaseSettings):
    REDIS_RATE_LIMIT_HOST: str = config("REDIS_RATE_LIMIT_HOST", default="localhost")
    REDIS_RATE_LIMIT_PORT: int = config("REDIS_RATE_LIMIT_PORT", default=6379)
    REDIS_RATE_LIMIT_URL: str = f"redis://{REDIS_RATE_LIMIT_HOST}:{REDIS_RATE_LIMIT_PORT}"


class DefaultRateLimitSettings(BaseSettings):
    DEFAULT_RATE_LIMIT_LIMIT: int = config("DEFAULT_RATE_LIMIT_LIMIT", default=10)
    DEFAULT_RATE_LIMIT_PERIOD: int = config("DEFAULT_RATE_LIMIT_PERIOD", default=3600)


class CRUDAdminSettings(BaseSettings):
    CRUD_ADMIN_ENABLED: bool = config("CRUD_ADMIN_ENABLED", default=True)
    CRUD_ADMIN_MOUNT_PATH: str = config("CRUD_ADMIN_MOUNT_PATH", default="/admin")

    CRUD_ADMIN_ALLOWED_IPS_LIST: list[str] | None = None
    CRUD_ADMIN_ALLOWED_NETWORKS_LIST: list[str] | None = None
    CRUD_ADMIN_MAX_SESSIONS: int = config("CRUD_ADMIN_MAX_SESSIONS", default=10)
    CRUD_ADMIN_SESSION_TIMEOUT: int = config("CRUD_ADMIN_SESSION_TIMEOUT", default=1440)
    SESSION_SECURE_COOKIES: bool = config("SESSION_SECURE_COOKIES", default=True)

    CRUD_ADMIN_TRACK_EVENTS: bool = config("CRUD_ADMIN_TRACK_EVENTS", default=True)
    CRUD_ADMIN_TRACK_SESSIONS: bool = config("CRUD_ADMIN_TRACK_SESSIONS", default=True)

    CRUD_ADMIN_REDIS_ENABLED: bool = config("CRUD_ADMIN_REDIS_ENABLED", default=False)
    CRUD_ADMIN_REDIS_HOST: str = config("CRUD_ADMIN_REDIS_HOST", default="localhost")
    CRUD_ADMIN_REDIS_PORT: int = config("CRUD_ADMIN_REDIS_PORT", default=6379)
    CRUD_ADMIN_REDIS_DB: int = config("CRUD_ADMIN_REDIS_DB", default=0)
    CRUD_ADMIN_REDIS_PASSWORD: str | None = config("CRUD_ADMIN_REDIS_PASSWORD", default="None")
    CRUD_ADMIN_REDIS_SSL: bool = config("CRUD_ADMIN_REDIS_SSL", default=False)


class VertexAISettings(BaseSettings):
    """Google Cloud Vertex AI Model Garden configuration for RAG system."""
    
    # Project settings
    PROJECT_ID: str = config("PROJECT_ID", default="g-72-courtsightteam")
    LOCATION: str = config("LOCATION", default="us-central1")
    
    # Model Garden endpoints
    EMBEDDING_SERVICE_URL: str = config("EMBEDDING_SERVICE_URL")
    EMBEDDING_SERVICE_ENDPOINT: str = config("EMBEDDING_SERVICE_ENDPOINT")
    LLM_SERVICE_URL: str = config("LLM_SERVICE_URL")
    LLM_SERVICE_ENDPOINT: str = config("LLM_SERVICE_ENDPOINT")
    
    # Model settings
    EMBEDDING_MODEL_NAME: str = config("EMBEDDING_MODEL_NAME", default="textembedding-gecko@003")
    LLM_MODEL_NAME: str = config("LLM_MODEL_NAME", default="chat-bison@002")
    
    # Authentication
    GCLOUD_TOKEN: SecretStr = config("GCLOUD_TOKEN", cast=SecretStr)
    GOOGLE_APPLICATION_CREDENTIALS: str | None = config("GOOGLE_APPLICATION_CREDENTIALS", default=None)
    GOOGLE_API_KEY: SecretStr = config("GOOGLE_API_KEY", cast=SecretStr, default="AIzaSyCBhxBMhS1Oe0S2NN4immDZdetZVwmLfy8")
    
    # API Keys for services
    EMBEDDING_SERVICE_API_KEY: str = config("EMBEDDING_SERVICE_API_KEY", default="embedding-service-dev-key")
    LLM_SERVICE_API_KEY: str = config("LLM_SERVICE_API_KEY", default="llm-service-dev-key")


class RAGSettings(BaseSettings):
    """RAG system configuration following PRD specifications."""
    
    # Vector database settings
    DATABASE_URL: str = config("POSTGRES_URL")
    VECTOR_COLLECTION_NAME: str = config("VECTOR_COLLECTION_NAME", default="supreme_court_docs")
    
    # Chunking settings (PRD specifications)
    PARENT_CHUNK_SIZE: int = config("PARENT_CHUNK_SIZE", default=2000)
    PARENT_CHUNK_OVERLAP: int = config("PARENT_CHUNK_OVERLAP", default=200)
    CHILD_CHUNK_SIZE: int = config("CHILD_CHUNK_SIZE", default=400)
    CHILD_CHUNK_OVERLAP: int = config("CHILD_CHUNK_OVERLAP", default=50)
    
    # Search settings
    MAX_SEARCH_RESULTS: int = config("MAX_SEARCH_RESULTS", default=10)
    SIMILARITY_THRESHOLD: float = config("SIMILARITY_THRESHOLD", default=0.7)
    SEARCH_TIMEOUT_SECONDS: int = config("SEARCH_TIMEOUT_SECONDS", default=10)
    
    # LLM settings
    LLM_TEMPERATURE: float = config("LLM_TEMPERATURE", default=0.1)
    LLM_MAX_TOKENS: int = config("LLM_MAX_TOKENS", default=2048)
    
    # Validation settings
    ENABLE_CLAIM_VALIDATION: bool = config("ENABLE_CLAIM_VALIDATION", default=True)
    VALIDATION_CONFIDENCE_THRESHOLD: float = config("VALIDATION_CONFIDENCE_THRESHOLD", default=0.7)
    
    # Performance settings
    ENABLE_CACHING: bool = config("ENABLE_CACHING", default=True)
    CACHE_TTL_SECONDS: int = config("CACHE_TTL_SECONDS", default=3600)
    BATCH_PROCESSING_SIZE: int = config("BATCH_PROCESSING_SIZE", default=50)


class EvaluationSettings(BaseSettings):
    """RAGAS evaluation and monitoring configuration."""
    
    # RAGAS evaluation
    ENABLE_RAGAS_EVALUATION: bool = config("ENABLE_RAGAS_EVALUATION", default=True)
    EVALUATION_BATCH_SIZE: int = config("EVALUATION_BATCH_SIZE", default=10)
    
    # Logging
    LOG_LEVEL: str = config("LOG_LEVEL", default="INFO")
    LOG_FORMAT: str = config("LOG_FORMAT", default="json")
    
    # Metrics collection
    ENABLE_METRICS: bool = config("ENABLE_METRICS", default=True)
    METRICS_PORT: int = config("METRICS_PORT", default=8080)
    
    # Health checks
    HEALTH_CHECK_TIMEOUT: int = config("HEALTH_CHECK_TIMEOUT", default=5)




class EnvironmentOption(Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class EnvironmentSettings(BaseSettings):
    ENVIRONMENT: EnvironmentOption = config("ENVIRONMENT", default=EnvironmentOption.LOCAL)


class Settings(
    AppSettings,
    PostgresSettings,
    CryptSettings,
    FirstUserSettings,
    TestSettings,
    RedisCacheSettings,
    ClientSideCacheSettings,
    RedisQueueSettings,
    RedisRateLimiterSettings,
    DefaultRateLimitSettings,
    CRUDAdminSettings,
    VertexAISettings,
    RAGSettings,
    EvaluationSettings,
    EnvironmentSettings,
):
    """Main application settings combining all configuration sections."""
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == EnvironmentOption.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == EnvironmentOption.LOCAL



    
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings