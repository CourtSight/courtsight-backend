from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass

from ..config import settings


class Base(DeclarativeBase, MappedAsDataclass):
    pass


# Import legal models after Base is defined to register them
def _register_legal_models():
    """Register legal models with main Base for Alembic detection."""
    from ...models.legal_document import LegalDocumentBase, LegalDocument, DocumentCitation, SearchQuery
    
    # Add legal model tables to main Base metadata
    for table in LegalDocumentBase.metadata.tables.values():
        # Add each table to main Base metadata if not already present
        if table.name not in Base.metadata.tables:
            table.tometadata(Base.metadata)

# Register legal models for Alembic
_register_legal_models()


DATABASE_URI = settings.POSTGRES_URI
DATABASE_PREFIX = settings.POSTGRES_ASYNC_PREFIX
DATABASE_URL = f"{DATABASE_PREFIX}{DATABASE_URI}"

async_engine = create_async_engine(DATABASE_URL, echo=False, future=True)

local_session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)


async def async_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with local_session() as db:
        yield db
