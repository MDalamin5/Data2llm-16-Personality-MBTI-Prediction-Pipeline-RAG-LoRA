"""Add updated_at triggers

Revision ID: 12604f430c01
Revises: 83cd04ac55b9
Create Date: 2025-10-14 14:10:56.569994

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = '12604f430c01'
down_revision: Union[str, Sequence[str], None] = '83cd04ac55b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
