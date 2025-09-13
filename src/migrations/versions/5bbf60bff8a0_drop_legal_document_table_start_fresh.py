"""drop_legal_document_table_start_fresh

Revision ID: 5bbf60bff8a0
Revises: f6395f5971f2
Create Date: 2025-08-31 16:50:01.969719

"""
from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '16eb520bfea3'
down_revision: Union[str, None] = 'e169d1834abd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop indexes first
    op.drop_index('idx_legal_document_active_with_embedding', table_name='legal_document')
    op.drop_index('idx_legal_document_embedding_id', table_name='legal_document')
    op.drop_index('idx_legal_document_collection_id', table_name='legal_document')
    op.drop_index('idx_legal_document_uuid', table_name='legal_document')
    op.drop_index('idx_legal_document_jurisdiction', table_name='legal_document')
    op.drop_index('idx_legal_document_is_active', table_name='legal_document')
    op.drop_index('idx_legal_document_court_name', table_name='legal_document')
    op.drop_index('idx_legal_document_content_hash', table_name='legal_document')
    op.drop_index('idx_legal_document_case_number', table_name='legal_document')
    op.drop_index('idx_legal_document_jurisdiction_type', table_name='legal_document')
    op.drop_index('idx_legal_document_date', table_name='legal_document')

    # Drop indexes for document_citation
    op.drop_index('idx_citation_case_number', table_name='document_citation')
    op.drop_index('ix_document_citation_cited_case_number', table_name='document_citation')
    op.drop_index('ix_document_citation_document_id', table_name='document_citation')
    op.drop_index('ix_document_citation_is_validated', table_name='document_citation')

    # Drop foreign key constraints
    op.drop_constraint('fk_legal_document_embedding_id', 'legal_document', type_='foreignkey')
    op.drop_constraint('fk_legal_document_collection_id', 'legal_document', type_='foreignkey')

    # Drop tables (document_citation first due to foreign key)
    op.drop_table('document_citation')
    op.drop_table('legal_document')

    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS match_legal_documents(vector, float, int, text, text, text, timestamptz, timestamptz)")
    op.execute("DROP FUNCTION IF EXISTS get_legal_document_statistics()")


def downgrade() -> None:
    # This migration is for starting fresh, so downgrade would recreate the tables
    # For now, we'll leave this empty as the user wants to start from scratch
    pass
