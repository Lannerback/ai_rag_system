"""init pgvector schema

Revision ID: 0001
Revises:
Create Date: 2026-06-29

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "embedding_collections",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("name", name="uq_embedding_collections_name"),
    )

    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("collection_id", UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("checksum", sa.String(), nullable=False),
        sa.Column("metadata", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["collection_id"], ["embedding_collections.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("collection_id", "source", name="uq_documents_collection_source"),
    )
    op.create_index("ix_documents_collection_id", "documents", ["collection_id"])

    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), nullable=False),
        sa.Column("collection_id", UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("language", sa.String(), nullable=True),
        sa.Column("loader", sa.String(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("checksum", sa.String(), nullable=False),
        sa.Column("metadata", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["collection_id"], ["embedding_collections.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        sa.CheckConstraint(
            "vector_dims(embedding) = embedding_dimensions", name="ck_chunks_embedding_dims"
        ),
    )
    op.create_index("ix_chunks_collection_id", "chunks", ["collection_id"])
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index(
        "ix_chunks_metadata_gin", "chunks", ["metadata"], postgresql_using="gin"
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_metadata_gin", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_index("ix_chunks_collection_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_documents_collection_id", table_name="documents")
    op.drop_table("documents")
    op.drop_table("embedding_collections")
