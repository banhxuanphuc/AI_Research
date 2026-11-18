"""paper card evidence report and usage

Revision ID: 6db356e96287
Revises: f6a4dfe94821
Create Date: 2026-11-18 17:04:08.424963
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '6db356e96287'
down_revision: str | None = 'f6a4dfe94821'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('paper_cards', sa.Column('evidence_report', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False))
    op.add_column('paper_cards', sa.Column('input_tokens', sa.Integer(), server_default='0', nullable=False))
    op.add_column('paper_cards', sa.Column('output_tokens', sa.Integer(), server_default='0', nullable=False))
    op.add_column('paper_cards', sa.Column('cost_usd', sa.Float(), server_default='0', nullable=False))
    op.add_column('paper_cards', sa.Column('latency_ms', sa.Integer(), server_default='0', nullable=False))


def downgrade() -> None:
    op.drop_column('paper_cards', 'latency_ms')
    op.drop_column('paper_cards', 'cost_usd')
    op.drop_column('paper_cards', 'output_tokens')
    op.drop_column('paper_cards', 'input_tokens')
    op.drop_column('paper_cards', 'evidence_report')
