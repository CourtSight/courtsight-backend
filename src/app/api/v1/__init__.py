from fastapi import APIRouter

from .login import router as login_router
from .logout import router as logout_router
from .rate_limits import router as rate_limits_router
from .tiers import router as tiers_router
from .users import router as users_router

# from .legal_search import router as legal_search_router  # Commented out - file doesn't exist
# from .document_processing import router as document_processing_router

router = APIRouter(prefix="/v1")
router.include_router(login_router)
router.include_router(logout_router)
router.include_router(users_router)
router.include_router(tiers_router)
router.include_router(rate_limits_router)
# router.include_router(legal_search_router)  # Commented out - file doesn't exist
# router.include_router(document_processing_router)
