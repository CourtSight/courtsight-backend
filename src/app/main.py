from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .admin.initialize import create_admin_interface
from .api import router
from .api.routes.search import router as search_router
from .api.routes.retrieval import router as retrieval_router
from .core.config import settings
from .core.setup import create_application, lifespan_factory
from .core.dependencies import startup_event, shutdown_event

admin = create_admin_interface()


@asynccontextmanager
async def lifespan_with_admin_and_rag(app: FastAPI) -> AsyncGenerator[None, None]:
    """Custom lifespan that includes admin and RAG system initialization."""
    # Get the default lifespan
    default_lifespan = lifespan_factory(settings)

    # Run the default lifespan initialization and our custom initialization
    async with default_lifespan(app):
        # Initialize RAG system
        await startup_event()
        
        # Initialize admin interface if it exists
        if admin:
            # Initialize admin database and setup
            await admin.initialize()

        yield
        
        # Cleanup RAG system
        await shutdown_event()


app = create_application(
    router=router, 
    settings=settings, 
    lifespan=lifespan_with_admin_and_rag
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include RAG search routes
app.include_router(search_router)

# Include enhanced retrieval routes (already has /api/v1/retrieval prefix)
app.include_router(retrieval_router)

# Mount admin interface if enabled
if admin:
    app.mount(settings.CRUD_ADMIN_MOUNT_PATH, admin.app)


# Add health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    from .core.dependencies import check_service_health
    return await check_service_health()


# Add metrics endpoint if enabled
if settings.ENABLE_METRICS:
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus-style metrics endpoint."""
        from .core.dependencies import get_metrics_collector
        collector = get_metrics_collector()
        # Return metrics in Prometheus format
        return {"status": "metrics_enabled"}


# Root endpoint with API information
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "AI-Powered Supreme Court Document Search Engine",
        "environment": settings.ENVIRONMENT.value,
        "endpoints": {
            "search": "/api/v1/search",
            "retrieval": "/api/v1/retrieval",
            "health": "/health",
            "docs": "/docs",
            "admin": settings.CRUD_ADMIN_MOUNT_PATH if admin else None
        }
    }
