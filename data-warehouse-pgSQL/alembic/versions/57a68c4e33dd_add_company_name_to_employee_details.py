"""add company_name to employee_details

Revision ID: 57a68c4e33dd
Revises: f6d4d37787b6
Create Date: 2025-10-25 10:15:17.292616

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = '57a68c4e33dd'
down_revision: Union[str, Sequence[str], None] = 'f6d4d37787b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add company_name column to employee_details table."""
    op.add_column(
        "employee_details",
        sa.Column("company_name", sa.String(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema: remove company_name column from employee_details table."""
    op.drop_column("employee_details", "company_name")
