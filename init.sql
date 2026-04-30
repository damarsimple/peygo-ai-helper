-- Create the ADK database (separate from main pelgo DB)
SELECT 'CREATE DATABASE pelgo_adk'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'pelgo_adk')\gexec

-- Seed ADK schema version so DatabaseSessionService initializes correctly.
-- Must run AFTER the pelgo_adk database is created.
\connect pelgo_adk
CREATE TABLE IF NOT EXISTS adk_internal_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR
);
INSERT INTO adk_internal_metadata (key, value)
VALUES ('schema_version', '1')
ON CONFLICT DO NOTHING;
