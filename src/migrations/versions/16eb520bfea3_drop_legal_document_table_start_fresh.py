"""drop_legal_document_table_start_fresh

Revision ID: 5bbf60bff8a0
Revises: f6395f5971f2
Create Date: 2025-08-31 16:50:01.969719

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '16eb520bfea3'
down_revision: Union[str, None] = 'e169d1834abd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop indexes that exist (using IF EXISTS or checking)
    op.execute("DROP INDEX IF EXISTS idx_legal_document_active_with_embedding")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_embedding_id")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_collection_id")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_uuid")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_jurisdiction")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_is_active")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_court_name")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_content_hash")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_case_number")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_jurisdiction_type")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_date")
    
    # Drop indexes for document_citation
    op.execute("DROP INDEX IF EXISTS idx_citation_case_number")
    op.execute("DROP INDEX IF EXISTS ix_document_citation_cited_case_number")
    op.execute("DROP INDEX IF EXISTS ix_document_citation_document_id")
    op.execute("DROP INDEX IF EXISTS ix_document_citation_is_validated")
    
    # Drop foreign key constraints (if they exist)
    op.execute("ALTER TABLE legal_document DROP CONSTRAINT IF EXISTS fk_legal_document_embedding_id")
    op.execute("ALTER TABLE legal_document DROP CONSTRAINT IF EXISTS fk_legal_document_collection_id")
    
    # Drop tables (document_citation first due to foreign key)
    op.execute("DROP TABLE IF EXISTS document_citation")
    op.execute("DROP TABLE IF EXISTS legal_document")
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS match_legal_documents(vector, float, int, text, text, text, timestamptz, timestamptz)")
    op.execute("DROP FUNCTION IF EXISTS get_legal_document_statistics()")


def downgrade() -> None:
    # This migration is for starting fresh, so downgrade would recreate the tables
    # For now, we'll leave this empty as the user wants to start from scratch
    pass
