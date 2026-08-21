-- PP-3B isolated PostgreSQL bootstrap for TOOLS-ITERATIVE-SQL-INVESTIGATION.
-- Local disposable credentials only — not for production.

CREATE SCHEMA IF NOT EXISTS proof;

CREATE TABLE IF NOT EXISTS proof.parcel_events (
    parcel_id BIGINT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    region TEXT NOT NULL,
    origin_hub TEXT NOT NULL,
    destination_hub TEXT NOT NULL,
    carrier TEXT NOT NULL,
    service_type TEXT NOT NULL,
    route_type TEXT NOT NULL,
    distance_km NUMERIC(8, 2) NOT NULL,
    weight_kg NUMERIC(8, 2) NOT NULL,
    planned_hours NUMERIC(8, 2) NOT NULL,
    actual_hours NUMERIC(8, 2) NOT NULL,
    delayed BOOLEAN NOT NULL,
    weekday SMALLINT NOT NULL
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'proof_runtime') THEN
        CREATE ROLE proof_runtime LOGIN PASSWORD 'proof_runtime_local';
    END IF;
END
$$;

ALTER DATABASE iterative_sql_proof SET statement_timeout = '5s';
ALTER ROLE proof_runtime SET statement_timeout = '5s';

REVOKE ALL ON DATABASE iterative_sql_proof FROM PUBLIC;
GRANT CONNECT ON DATABASE iterative_sql_proof TO proof_runtime;

REVOKE ALL ON SCHEMA proof FROM PUBLIC;
GRANT USAGE ON SCHEMA proof TO proof_runtime;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON ALL TABLES IN SCHEMA proof FROM PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA proof TO proof_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA proof REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA proof GRANT SELECT ON TABLES TO proof_runtime;
