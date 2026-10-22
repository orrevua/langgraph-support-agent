"""Postgres implementation of `BankRepository` (psycopg 3 + a small connection pool).

Plain SQL on purpose: the queries are short and you can see exactly what hits the database.
"""

from __future__ import annotations

from importlib import resources

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from supportagent.domain import seed
from supportagent.domain.models import (
    Account,
    ApprovalRecord,
    Card,
    CardStatus,
    Customer,
    Refund,
    Transaction,
    TxnStatus,
)

TABLES = ["approval_audit", "refunds", "transactions", "cards", "accounts", "customers"]


class PostgresBankRepository:
    def __init__(self, database_url: str, *, pool: ConnectionPool | None = None) -> None:
        self.pool = pool or ConnectionPool(
            database_url, min_size=1, max_size=5, kwargs={"row_factory": dict_row}, open=True
        )

    def close(self) -> None:
        self.pool.close()

    # --- schema / seed -----------------------------------------------------------------
    def create_schema(self) -> None:
        sql = resources.files("supportagent.repository").joinpath("schema.sql").read_text()
        with self.pool.connection() as conn:
            conn.execute(sql)

    def reset_and_seed(self) -> None:
        """Drop all domain rows and insert the deterministic seed (idempotent)."""
        self.create_schema()
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(f"TRUNCATE {', '.join(TABLES)} CASCADE")
            conn.execute("ALTER SEQUENCE refund_seq RESTART WITH 100")
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO customers (id, full_name, cpf, email, phone, tier) "
                    "VALUES (%(id)s, %(full_name)s, %(cpf)s, %(email)s, %(phone)s, %(tier)s)",
                    [c.model_dump() for c in seed.CUSTOMERS],
                )
                cur.executemany(
                    "INSERT INTO accounts (id, customer_id, kind, balance_cents) "
                    "VALUES (%(id)s, %(customer_id)s, %(kind)s, %(balance_cents)s)",
                    [a.model_dump() for a in seed.ACCOUNTS],
                )
                cur.executemany(
                    "INSERT INTO cards (id, customer_id, kind, last4, status) "
                    "VALUES (%(id)s, %(customer_id)s, %(kind)s, %(last4)s, %(status)s)",
                    [c.model_dump() for c in seed.CARDS],
                )
                cur.executemany(
                    "INSERT INTO transactions (id, account_id, customer_id, kind, amount_cents, counterparty, "
                    "description, status, created_at, card_id, end_to_end_id, failure_code, failure_reason, "
                    "funds_returned) VALUES (%(id)s, %(account_id)s, %(customer_id)s, %(kind)s, "
                    "%(amount_cents)s, %(counterparty)s, %(description)s, %(status)s, %(created_at)s, "
                    "%(card_id)s, %(end_to_end_id)s, %(failure_code)s, %(failure_reason)s, %(funds_returned)s)",
                    [t.model_dump() for t in seed.TRANSACTIONS],
                )
                cur.executemany(
                    "INSERT INTO refunds (id, transaction_id, customer_id, amount_cents, reason, status, "
                    "created_at, approved_by) VALUES (%(id)s, %(transaction_id)s, %(customer_id)s, "
                    "%(amount_cents)s, %(reason)s, %(status)s, %(created_at)s, %(approved_by)s)",
                    [r.model_dump() for r in seed.REFUNDS],
                )

    # --- queries -----------------------------------------------------------------------
    def _one(self, sql: str, params: tuple) -> dict | None:
        with self.pool.connection() as conn:
            return conn.execute(sql, params).fetchone()

    def _all(self, sql: str, params: tuple) -> list[dict]:
        with self.pool.connection() as conn:
            return conn.execute(sql, params).fetchall()

    def get_customer(self, customer_id: str) -> Customer | None:
        row = self._one("SELECT * FROM customers WHERE id = %s", (customer_id,))
        return Customer(**row) if row else None

    def list_accounts(self, customer_id: str) -> list[Account]:
        rows = self._all("SELECT * FROM accounts WHERE customer_id = %s ORDER BY id", (customer_id,))
        return [Account(**r) for r in rows]

    def list_cards(self, customer_id: str) -> list[Card]:
        rows = self._all("SELECT * FROM cards WHERE customer_id = %s ORDER BY id", (customer_id,))
        return [Card(**r) for r in rows]

    def get_card(self, card_id: str) -> Card | None:
        row = self._one("SELECT * FROM cards WHERE id = %s", (card_id,))
        return Card(**row) if row else None

    def set_card_status(self, card_id: str, status: CardStatus) -> Card:
        row = self._one("UPDATE cards SET status = %s WHERE id = %s RETURNING *", (status, card_id))
        if row is None:
            raise KeyError(card_id)
        return Card(**row)

    def list_transactions(
        self, customer_id: str, *, limit: int = 10, status: TxnStatus | None = None
    ) -> list[Transaction]:
        rows = self._all(
            "SELECT * FROM transactions WHERE customer_id = %s AND (%s::text IS NULL OR status = %s) "
            "ORDER BY created_at DESC LIMIT %s",
            (customer_id, status, status, limit),
        )
        return [Transaction(**r) for r in rows]

    def get_transaction(self, transaction_id: str) -> Transaction | None:
        row = self._one("SELECT * FROM transactions WHERE id = %s", (transaction_id,))
        return Transaction(**row) if row else None

    def list_refunds(self, transaction_id: str) -> list[Refund]:
        rows = self._all("SELECT * FROM refunds WHERE transaction_id = %s ORDER BY created_at", (transaction_id,))
        return [Refund(**r) for r in rows]

    def create_refund(
        self,
        *,
        transaction_id: str,
        customer_id: str,
        amount_cents: int,
        reason: str,
        approved_by: str | None,
    ) -> Refund:
        # One transaction: the refund row and the balance credit succeed or fail together.
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                "INSERT INTO refunds (id, transaction_id, customer_id, amount_cents, reason, approved_by) "
                "VALUES ('RF-' || nextval('refund_seq'), %s, %s, %s, %s, %s) RETURNING *",
                (transaction_id, customer_id, amount_cents, reason, approved_by),
            ).fetchone()
            conn.execute(
                "UPDATE accounts SET balance_cents = balance_cents + %s "
                "WHERE id = (SELECT account_id FROM transactions WHERE id = %s)",
                (amount_cents, transaction_id),
            )
        return Refund(**row)

    def record_approval(self, record: ApprovalRecord) -> ApprovalRecord:
        # ON CONFLICT makes the write idempotent if a node is ever re-executed.
        row = self._one(
            "INSERT INTO approval_audit (thread_id, tool_call_id, tool_name, proposed_args, final_args, decision, "
            "reviewer, reason) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (thread_id, tool_call_id) DO UPDATE SET thread_id = EXCLUDED.thread_id RETURNING *",
            (
                record.thread_id,
                record.tool_call_id,
                record.tool_name,
                Jsonb(record.proposed_args),
                Jsonb(record.final_args) if record.final_args is not None else None,
                record.decision,
                record.reviewer,
                record.reason,
            ),
        )
        return ApprovalRecord(**row)

    def list_approvals(self, thread_id: str | None = None) -> list[ApprovalRecord]:
        rows = self._all(
            "SELECT * FROM approval_audit WHERE %s::text IS NULL OR thread_id = %s ORDER BY id",
            (thread_id, thread_id),
        )
        return [ApprovalRecord(**r) for r in rows]
