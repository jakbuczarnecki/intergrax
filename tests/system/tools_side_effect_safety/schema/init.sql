CREATE TABLE IF NOT EXISTS effects (
    id BIGSERIAL PRIMARY KEY,
    business_operation_id TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    worker_source TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_effects_business_operation_id
    ON effects (business_operation_id);

CREATE TABLE IF NOT EXISTS effect_attempts (
    id BIGSERIAL PRIMARY KEY,
    business_operation_id TEXT NOT NULL,
    request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    worker_source TEXT,
    proof_mode TEXT,
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_effect_attempts_business_operation_id
    ON effect_attempts (business_operation_id);
