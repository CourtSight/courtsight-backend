"""integrate_legal_document_with_langchain_pgstore

Revision ID: f6395f5971f2
Revises: e169d1834abd
Create Date: 2025-08-31 16:48:29.626395

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f6395f5971f2'
down_revision: Union[str, None] = 'e169d1834abd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


"""integrate_legal_document_with_langchain_pgstore

Revision ID: f6395f5971f2
Revises: e169d1834abd
Create Date: 2025-08-31 16:48:29.626395

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'f6395f5971f2'
down_revision: Union[str, None] = 'e169d1834abd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add collection_id column to link with langchain_pg_collection
    op.add_column('legal_document', sa.Column('collection_id', postgresql.UUID(), nullable=True))
    
    # Add embedding_id column to optionally link with langchain_pg_embedding
    op.add_column('legal_document', sa.Column('embedding_id', sa.String(), nullable=True))
    
    # Create a default collection for legal documents
    op.execute("""
        INSERT INTO langchain_pg_collection (uuid, name, cmetadata)
        VALUES (gen_random_uuid(), 'legal_documents', '{"description": "Legal court documents collection", "created_by": "migration"}')
        ON CONFLICT (name) DO NOTHING
    """)
    
    # Update existing records to use the default collection
    op.execute("""
        UPDATE legal_document 
        SET collection_id = (
            SELECT uuid FROM langchain_pg_collection 
            WHERE name = 'legal_documents' 
            LIMIT 1
        )
        WHERE collection_id IS NULL
    """)
    
    # Add foreign key constraints
    op.create_foreign_key(
        'fk_legal_document_collection_id',
        'legal_document', 'langchain_pg_collection',
        ['collection_id'], ['uuid'],
        ondelete='SET NULL'
    )
    
    op.create_foreign_key(
        'fk_legal_document_embedding_id',
        'legal_document', 'langchain_pg_embedding',
        ['embedding_id'], ['id'],
        ondelete='SET NULL'
    )
    
    # Add indexes for better performance
    op.create_index('idx_legal_document_collection_id', 'legal_document', ['collection_id'])
    op.create_index('idx_legal_document_embedding_id', 'legal_document', ['embedding_id'])
    
    # Add a partial index for active documents with embeddings
    op.execute("""
        CREATE INDEX idx_legal_document_active_with_embedding 
        ON legal_document (collection_id, embedding_id) 
        WHERE is_active = true AND embedding IS NOT NULL
    """)


def downgrade() -> None:
    # Remove indexes
    op.drop_index('idx_legal_document_active_with_embedding', table_name='legal_document')
    op.drop_index('idx_legal_document_embedding_id', table_name='legal_document')
    op.drop_index('idx_legal_document_collection_id', table_name='legal_document')
    
    # Remove foreign key constraints
    op.drop_constraint('fk_legal_document_embedding_id', 'legal_document', type_='foreignkey')
    op.drop_constraint('fk_legal_document_collection_id', 'legal_document', type_='foreignkey')
    
    # Remove columns
    op.drop_column('legal_document', 'embedding_id')
    op.drop_column('legal_document', 'collection_id')
    
    # Note: We don't remove the default collection as it might be used by other parts of the system
