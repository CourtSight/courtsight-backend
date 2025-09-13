from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .admin.initialize import create_admin_interface
from .api import router
from .api.routes.retrieval import router as retrieval_router
from .api.routes.search import router as search_router
from .core.config import settings
from .core.database import database_lifecycle, get_db_status
from .core.dependencies import shutdown_event, startup_event
from .core.setup import create_application, lifespan_factory

admin = create_admin_interface()


@asynccontextmanager
async def lifespan_with_admin_and_rag(app: FastAPI) -> AsyncGenerator[None, None]:
    """Custom lifespan that includes admin, RAG system, and database connection management."""
    # Get the default lifespan
    default_lifespan = lifespan_factory(settings)

    # Run the default lifespan initialization and our custom initialization
    async with default_lifespan(app):
        # Initialize database connections with lifecycle management
        async with database_lifecycle():
            # Initialize RAG system
            await startup_event()

            # Initialize admin interface if it exists
            if admin:
                # Initialize admin database and setup
                await admin.initialize()

            # Log database connection status
            db_status = get_db_status()
            print(f"🔗 Database connections initialized: {db_status['active_connections']}/{db_status['max_connections']}")

            yield

            # Cleanup RAG system
            await shutdown_event()
            print("🔗 Database connections will be closed by context manager")


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
    """Basic health check endpoint with database connection status."""
    from .core.dependencies import check_service_health

    # Get database connection status
    db_status = get_db_status()

    health_result = await check_service_health()
    health_result["database_connections"] = db_status

    return health_result


# Add database status endpoint
@app.get("/db-status")
async def database_status():
    """Database connection status endpoint."""
    return get_db_status()


# Add metrics endpoint if enabled
if settings.ENABLE_METRICS:
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus-style metrics endpoint."""
        from .core.dependencies import get_metrics_collector
        get_metrics_collector()
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
