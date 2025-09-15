"""Create parent_documents table

Revision ID: create_parent_documents
Revises: 
Create Date: 2025-09-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = 'create_parent_documents'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create parent_documents table for storing parent documents in ParentDocumentRetriever."""
    op.create_table(
        'parent_documents',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', JSONB(), nullable=True),
        sa.Column('collection_name', sa.String(255), nullable=False, index=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    # Create index for faster queries
    op.create_index('idx_parent_docs_collection', 'parent_documents', ['collection_name'])


def downgrade() -> None:
    """Drop parent_documents table."""
    op.drop_table('parent_documents')
