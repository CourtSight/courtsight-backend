"""add_supabase_vector_functions_and_indexes

Revision ID: e169d1834abd
Revises: 6441b28b335c
Create Date: 2025-08-12 12:21:59.115614

"""
from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e169d1834abd'
down_revision: Union[str, None] = '6441b28b335c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add Supabase pgvector functions and performance indexes for legal document search."""

    # Enable the pgvector extension (if not already enabled)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Function to match legal documents based on vector similarity
    op.execute("""
        CREATE OR REPLACE FUNCTION match_legal_documents(
          query_embedding vector(384),
          match_threshold float DEFAULT 0.7,
          match_count int DEFAULT 10,
          filter_jurisdiction text DEFAULT NULL,
          filter_case_type text DEFAULT NULL,
          filter_legal_area text DEFAULT NULL,
          filter_date_from timestamptz DEFAULT NULL,
          filter_date_to timestamptz DEFAULT NULL
        )
        RETURNS TABLE (
          id bigint,
          uuid uuid,
          case_number text,
          court_name text,
          jurisdiction text,
          title text,
          summary text,
          decision_date timestamptz,
          case_type text,
          legal_area text,
          similarity float
        )
        LANGUAGE sql STABLE
        AS $$
          SELECT
            ld.id,
            ld.uuid,
            ld.case_number,
            ld.court_name,
            ld.jurisdiction,
            ld.title,
            ld.summary,
            ld.decision_date,
            ld.case_type,
            ld.legal_area,
            1 - (ld.embedding <=> query_embedding) as similarity
          FROM legal_document ld
          WHERE ld.is_active = true
            AND ld.embedding IS NOT NULL
            AND 1 - (ld.embedding <=> query_embedding) > match_threshold
            -- Apply optional filters
            AND (filter_jurisdiction IS NULL OR ld.jurisdiction = filter_jurisdiction)
            AND (filter_case_type IS NULL OR ld.case_type = filter_case_type)
            AND (filter_legal_area IS NULL OR ld.legal_area = filter_legal_area)
            AND (filter_date_from IS NULL OR ld.decision_date >= filter_date_from)
            AND (filter_date_to IS NULL OR ld.decision_date <= filter_date_to)
          ORDER BY ld.embedding <=> query_embedding ASC
          LIMIT match_count;
        $$;
    """)

    # Function to get legal document statistics
    op.execute("""
        CREATE OR REPLACE FUNCTION get_legal_document_statistics()
        RETURNS jsonb
        LANGUAGE sql STABLE
        AS $$
          SELECT jsonb_build_object(
            'total_documents', COUNT(*),
            'active_documents', COUNT(*) FILTER (WHERE is_active = true),
            'documents_with_embeddings', COUNT(*) FILTER (WHERE embedding IS NOT NULL),
            'by_jurisdiction', (
              SELECT jsonb_object_agg(jurisdiction, doc_count)
              FROM (
                SELECT jurisdiction, COUNT(*) as doc_count
                FROM legal_document
                WHERE is_active = true
                GROUP BY jurisdiction
              ) jurisdiction_stats
            ),
            'by_case_type', (
              SELECT jsonb_object_agg(case_type, doc_count)
              FROM (
                SELECT case_type, COUNT(*) as doc_count
                FROM legal_document
                WHERE is_active = true AND case_type IS NOT NULL
                GROUP BY case_type
              ) case_type_stats
            ),
            'by_legal_area', (
              SELECT jsonb_object_agg(legal_area, doc_count)
              FROM (
                SELECT legal_area, COUNT(*) as doc_count
                FROM legal_document
                WHERE is_active = true AND legal_area IS NOT NULL
                GROUP BY legal_area
              ) legal_area_stats
            )
          )
          FROM legal_document;
        $$;
    """)

    # Function to get popular search queries
    op.execute("""
        CREATE OR REPLACE FUNCTION get_popular_search_queries(
          limit_count int DEFAULT 10,
          days_back int DEFAULT 30
        )
        RETURNS TABLE (
          query_text text,
          search_count bigint,
          avg_results_count numeric,
          avg_response_time_ms numeric
        )
        LANGUAGE sql STABLE
        AS $$
          SELECT
            sq.query_text,
            COUNT(*) as search_count,
            AVG(sq.results_count) as avg_results_count,
            AVG(sq.response_time_ms) as avg_response_time_ms
          FROM search_query sq
          WHERE sq.created_at >= NOW() - INTERVAL '1 day' * days_back
            AND sq.results_count > 0
          GROUP BY sq.query_text
          ORDER BY search_count DESC, avg_results_count DESC
          LIMIT limit_count;
        $$;
    """)

    # Create indexes for better performance
    # Vector index for similarity search (using HNSW algorithm)
    op.execute("""
        CREATE INDEX IF NOT EXISTS legal_document_embedding_idx
        ON legal_document
        USING hnsw (embedding vector_cosine_ops);
    """)

    # Additional indexes for filtering
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_legal_document_active_embedding
        ON legal_document (is_active, embedding)
        WHERE is_active = true AND embedding IS NOT NULL;
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_legal_document_jurisdiction_active
        ON legal_document (jurisdiction, is_active)
        WHERE is_active = true;
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_legal_document_case_type_active
        ON legal_document (case_type, is_active)
        WHERE is_active = true AND case_type IS NOT NULL;
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_legal_document_legal_area_active
        ON legal_document (legal_area, is_active)
        WHERE is_active = true AND legal_area IS NOT NULL;
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_legal_document_decision_date_active
        ON legal_document (decision_date, is_active)
        WHERE is_active = true AND decision_date IS NOT NULL;
    """)

    # Composite index for common filter combinations
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_legal_document_filters
        ON legal_document (jurisdiction, case_type, legal_area, decision_date, is_active)
        WHERE is_active = true;
    """)

    # Search query analytics indexes
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_search_query_created_at
        ON search_query (created_at DESC);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_search_query_text_results
        ON search_query (query_text, results_count)
        WHERE results_count > 0;
    """)

    # Add comments for documentation
    op.execute("COMMENT ON FUNCTION match_legal_documents IS 'Performs vector similarity search on legal documents with optional filtering';")
    op.execute("COMMENT ON FUNCTION get_legal_document_statistics IS 'Returns comprehensive statistics about the legal document collection';")
    op.execute("COMMENT ON FUNCTION get_popular_search_queries IS 'Returns popular search queries with performance metrics';")
    op.execute("COMMENT ON INDEX legal_document_embedding_idx IS 'HNSW index for fast vector similarity search using cosine distance';")


def downgrade() -> None:
    """Remove Supabase pgvector functions and performance indexes."""

    # Drop indexes
    op.execute("DROP INDEX IF EXISTS idx_search_query_text_results;")
    op.execute("DROP INDEX IF EXISTS idx_search_query_created_at;")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_filters;")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_decision_date_active;")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_legal_area_active;")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_case_type_active;")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_jurisdiction_active;")
    op.execute("DROP INDEX IF EXISTS idx_legal_document_active_embedding;")
    op.execute("DROP INDEX IF EXISTS legal_document_embedding_idx;")

    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS get_popular_search_queries;")
    op.execute("DROP FUNCTION IF EXISTS get_legal_document_statistics;")
    op.execute("DROP FUNCTION IF EXISTS match_legal_documents;")
