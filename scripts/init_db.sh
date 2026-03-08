#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-'EOSQL'
    CREATE TABLE IF NOT EXISTS pdm_agent_memory (
        memory_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        equipment_id        VARCHAR(50) NOT NULL,
        bearing_id          VARCHAR(50) NOT NULL,
        event_id            VARCHAR(100) NOT NULL,
        event_timestamp     TIMESTAMP NOT NULL,
        created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        fault_type          VARCHAR(30),
        fault_stage         INTEGER,
        degradation_speed   VARCHAR(20),
        risk_level          VARCHAR(20),
        ml_rul_hours        FLOAT,
        agent_rul_assessment TEXT,
        recommendation      TEXT,
        uncertainty_notes   TEXT,
        reasoning_summary   TEXT,
        tools_used          JSONB,
        deep_research       BOOLEAN DEFAULT FALSE,
        human_response      TEXT,
        action_taken        TEXT,
        resolved            BOOLEAN DEFAULT FALSE
    );

    CREATE INDEX IF NOT EXISTS idx_memory_equipment_bearing
    ON pdm_agent_memory (equipment_id, bearing_id, event_timestamp DESC);
EOSQL
