-- Schema for the mock fintech (database p2_agent). Money is stored in integer cents.
-- LangGraph's PostgresSaver creates its own checkpoint tables via `saver.setup()`.

CREATE TABLE IF NOT EXISTS customers (
    id          text PRIMARY KEY,
    full_name   text NOT NULL,
    cpf         text NOT NULL UNIQUE,
    email       text NOT NULL UNIQUE,
    phone       text NOT NULL,
    tier        text NOT NULL DEFAULT 'standard' CHECK (tier IN ('standard', 'premium'))
);

CREATE TABLE IF NOT EXISTS accounts (
    id            text PRIMARY KEY,
    customer_id   text NOT NULL REFERENCES customers(id),
    kind          text NOT NULL CHECK (kind IN ('checking', 'savings')),
    balance_cents bigint NOT NULL
);

CREATE TABLE IF NOT EXISTS cards (
    id          text PRIMARY KEY,
    customer_id text NOT NULL REFERENCES customers(id),
    kind        text NOT NULL CHECK (kind IN ('debit', 'credit')),
    last4       char(4) NOT NULL,
    status      text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'blocked'))
);

CREATE TABLE IF NOT EXISTS transactions (
    id             text PRIMARY KEY,
    account_id     text NOT NULL REFERENCES accounts(id),
    customer_id    text NOT NULL REFERENCES customers(id),
    kind           text NOT NULL CHECK (kind IN ('pix_out', 'pix_in', 'card_purchase')),
    amount_cents   bigint NOT NULL CHECK (amount_cents > 0),
    counterparty   text NOT NULL,
    description    text NOT NULL,
    status         text NOT NULL CHECK (status IN ('completed', 'pending', 'failed', 'disputed')),
    created_at     timestamptz NOT NULL,
    card_id        text REFERENCES cards(id),
    end_to_end_id  text,
    failure_code   text,
    failure_reason text,
    funds_returned boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS transactions_customer_created_idx ON transactions (customer_id, created_at DESC);

CREATE TABLE IF NOT EXISTS refunds (
    id             text PRIMARY KEY,
    transaction_id text NOT NULL REFERENCES transactions(id),
    customer_id    text NOT NULL REFERENCES customers(id),
    amount_cents   bigint NOT NULL CHECK (amount_cents > 0),
    reason         text NOT NULL,
    status         text NOT NULL DEFAULT 'completed',
    created_at     timestamptz NOT NULL DEFAULT now(),
    approved_by    text
);
CREATE SEQUENCE IF NOT EXISTS refund_seq START 100;

-- Audit trail of human decisions on sensitive actions (append-only).
CREATE TABLE IF NOT EXISTS approval_audit (
    id            bigserial PRIMARY KEY,
    thread_id     text NOT NULL,
    tool_call_id  text NOT NULL,
    tool_name     text NOT NULL,
    proposed_args jsonb NOT NULL,
    final_args    jsonb,
    decision      text NOT NULL CHECK (decision IN ('approved', 'rejected')),
    reviewer      text NOT NULL,
    reason        text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (thread_id, tool_call_id)
);
CREATE INDEX IF NOT EXISTS approval_audit_thread_idx ON approval_audit (thread_id, created_at);
