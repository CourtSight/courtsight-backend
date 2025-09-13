from arq.connections import RedisSettings

from ...core.config import settings
from .functions import process_documents_background, shutdown, startup, track_search_analytics

REDIS_QUEUE_HOST = settings.REDIS_QUEUE_HOST
REDIS_QUEUE_PORT = settings.REDIS_QUEUE_PORT


class WorkerSettings:
    functions = [track_search_analytics, process_documents_background]  # Add your background tasks here
    redis_settings = RedisSettings(host=REDIS_QUEUE_HOST, port=REDIS_QUEUE_PORT)
    on_startup = startup
    on_shutdown = shutdown
    handle_signals = False
