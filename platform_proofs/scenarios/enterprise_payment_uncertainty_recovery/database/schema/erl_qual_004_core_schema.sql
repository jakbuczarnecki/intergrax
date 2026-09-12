-- ERL-QUAL-004 canonical schema snapshot (keep in sync with migrations/001_erl_qual_004_core_schema.sql)
-- Architecture: docs/ERL_QUAL_004_POSTGRESQL_DATA_MODEL_ARCHITECTURE.md

BEGIN;

CREATE SCHEMA IF NOT EXISTS commerce;
CREATE SCHEMA IF NOT EXISTS external_sor;
CREATE SCHEMA IF NOT EXISTS reconciliation;

CREATE TABLE commerce.organizations (
    organization_id uuid PRIMARY KEY,
    organization_key text NOT NULL,
    legal_name text NOT NULL,
    account_reference text,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    CONSTRAINT organizations_organization_key_key UNIQUE (organization_key)
);

CREATE TABLE commerce.orders (
    order_id uuid PRIMARY KEY,
    organization_id uuid NOT NULL REFERENCES commerce.organizations (organization_id),
    order_number text NOT NULL,
    amount numeric(18, 2) NOT NULL,
    currency char(3) NOT NULL,
    business_status text NOT NULL,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    fulfillment_eligible_at timestamptz,
    CONSTRAINT orders_order_number_key UNIQUE (order_number),
    CONSTRAINT orders_amount_positive CHECK (amount > 0)
);

CREATE TABLE commerce.payment_intents (
    payment_intent_id uuid PRIMARY KEY,
    order_id uuid NOT NULL REFERENCES commerce.orders (order_id),
    intent_reference text NOT NULL,
    amount numeric(18, 2) NOT NULL,
    currency char(3) NOT NULL,
    correlation_id text NOT NULL,
    attempt_ordinal integer NOT NULL DEFAULT 1,
    application_status text NOT NULL,
    requested_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    CONSTRAINT payment_intents_intent_reference_key UNIQUE (intent_reference),
    CONSTRAINT payment_intents_amount_positive CHECK (amount > 0),
    CONSTRAINT payment_intents_attempt_ordinal_positive CHECK (attempt_ordinal >= 1)
);

CREATE TABLE commerce.application_knowledge (
    application_knowledge_id uuid PRIMARY KEY,
    payment_intent_id uuid NOT NULL REFERENCES commerce.payment_intents (payment_intent_id),
    external_payment_effect_id uuid NOT NULL,
    known_status text NOT NULL,
    order_payment_substate text NOT NULL,
    confirmation_received boolean NOT NULL,
    uncertainty_explicit boolean NOT NULL,
    inventory_reservation_knowledge text,
    observed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    CONSTRAINT application_knowledge_known_status_check CHECK (
        known_status IN ('UNKNOWN', 'CONFIRMED', 'FAILED')
    ),
    CONSTRAINT application_knowledge_unknown_explicit_check CHECK (
        (known_status = 'UNKNOWN') = uncertainty_explicit
    ),
    CONSTRAINT application_knowledge_payment_intent_key UNIQUE (payment_intent_id)
);

CREATE TABLE external_sor.external_payment_effects (
    external_payment_effect_id uuid PRIMARY KEY,
    payment_intent_id uuid NOT NULL REFERENCES commerce.payment_intents (payment_intent_id),
    external_effect_reference text NOT NULL,
    correlation_id text NOT NULL,
    requested_state text NOT NULL,
    observed_integration_state text NOT NULL,
    requested_at timestamptz NOT NULL,
    observed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    CONSTRAINT external_payment_effects_reference_key UNIQUE (external_effect_reference)
);

ALTER TABLE commerce.application_knowledge
    ADD CONSTRAINT application_knowledge_external_effect_fkey
    FOREIGN KEY (external_payment_effect_id)
    REFERENCES external_sor.external_payment_effects (external_payment_effect_id);

CREATE TABLE external_sor.external_reality (
    external_reality_id uuid PRIMARY KEY,
    external_payment_effect_id uuid NOT NULL REFERENCES external_sor.external_payment_effects (
        external_payment_effect_id
    ),
    correlation_id text NOT NULL,
    sor_transaction_ref text,
    terminal_outcome text NOT NULL,
    funds_captured boolean NOT NULL,
    truth_availability_state text NOT NULL,
    processed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    CONSTRAINT external_reality_effect_key UNIQUE (external_payment_effect_id),
    CONSTRAINT external_reality_terminal_outcome_check CHECK (
        terminal_outcome IN (
            'PAYMENT_COMPLETED',
            'PAYMENT_FAILED',
            'TRUTH_INDETERMINATE'
        )
    ),
    CONSTRAINT external_reality_truth_availability_check CHECK (
        truth_availability_state IN ('AVAILABLE', 'UNAVAILABLE', 'INDETERMINATE')
    )
);

CREATE TABLE reconciliation.reconciliation_cases (
    reconciliation_case_id uuid PRIMARY KEY,
    case_reference text NOT NULL,
    order_id uuid NOT NULL REFERENCES commerce.orders (order_id),
    payment_intent_id uuid NOT NULL REFERENCES commerce.payment_intents (payment_intent_id),
    correlation_id text NOT NULL,
    variant_context text,
    resolution_state text NOT NULL,
    opened_at timestamptz NOT NULL,
    resolved_at timestamptz,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    CONSTRAINT reconciliation_cases_case_reference_key UNIQUE (case_reference),
    CONSTRAINT reconciliation_cases_resolution_state_check CHECK (
        resolution_state IN ('OPEN', 'RESOLVED', 'ESCALATED', 'CLOSED_UNRESOLVED')
    )
);

CREATE TABLE reconciliation.investigation_attempts (
    investigation_attempt_id uuid PRIMARY KEY,
    reconciliation_case_id uuid NOT NULL REFERENCES reconciliation.reconciliation_cases (
        reconciliation_case_id
    ),
    attempt_number integer NOT NULL,
    requested_at timestamptz NOT NULL,
    observed_at timestamptz,
    outcome text NOT NULL,
    summary_code text,
    created_at timestamptz NOT NULL,
    CONSTRAINT investigation_attempts_case_attempt_key UNIQUE (
        reconciliation_case_id,
        attempt_number
    ),
    CONSTRAINT investigation_attempts_attempt_positive CHECK (attempt_number >= 1),
    CONSTRAINT investigation_attempts_outcome_check CHECK (
        outcome IN ('SUCCESS', 'FAILED', 'UNAVAILABLE')
    )
);

CREATE TABLE reconciliation.evidence_references (
    evidence_reference_id uuid PRIMARY KEY,
    reconciliation_case_id uuid NOT NULL REFERENCES reconciliation.reconciliation_cases (
        reconciliation_case_id
    ),
    investigation_attempt_id uuid REFERENCES reconciliation.investigation_attempts (
        investigation_attempt_id
    ),
    reference_type text NOT NULL,
    external_reference text NOT NULL,
    created_at timestamptz NOT NULL
);

CREATE TABLE reconciliation.resolution_records (
    resolution_record_id uuid PRIMARY KEY,
    reconciliation_case_id uuid NOT NULL REFERENCES reconciliation.reconciliation_cases (
        reconciliation_case_id
    ),
    alignment_action text NOT NULL,
    recorded_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL,
    CONSTRAINT resolution_records_case_key UNIQUE (reconciliation_case_id),
    CONSTRAINT resolution_records_alignment_action_check CHECK (
        alignment_action IN (
            'KNOWLEDGE_ALIGNED_TO_SUCCEEDED',
            'KNOWLEDGE_ALIGNED_TO_FAILED',
            'ESCALATED_HUMAN',
            'CONTAINED_NO_TRUTH'
        )
    )
);

COMMIT;
