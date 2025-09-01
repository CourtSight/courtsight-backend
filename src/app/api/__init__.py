from fastapi import APIRouter

from .v1 import router as v1_router
from .routes.retrieval import router as retrieval_router

router = APIRouter(prefix="/api")
router.include_router(v1_router)
router.include_router(retrieval_router)
