-- Create the ADK database (separate from main pelgo DB)
SELECT 'CREATE DATABASE pelgo_adk'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'pelgo_adk')\gexec
