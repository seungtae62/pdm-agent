"""Qdrant collection initialization script.

Creates required collections if they don't already exist (idempotent).
Run manually: python scripts/init_qdrant.py
"""

import os

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTIONS: list[dict] = [
    {
        "name": "maintenance_history",
        "vector_size": 1536,
        "payload_indexes": ["equipment_id", "bearing_id", "fault_type"],
    },
    {
        "name": "equipment_manual",
        "vector_size": 1536,
        "payload_indexes": ["doc_type", "equipment_model"],
    },
    {
        "name": "analysis_history",
        "vector_size": 1536,
        "payload_indexes": [
            "equipment_id",
            "bearing_id",
            "fault_type",
            "risk_level",
        ],
    },
]


def main() -> None:
    """Create Qdrant collections and payload indexes."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    for col in COLLECTIONS:
        name: str = col["name"]

        # Check if collection already exists
        try:
            client.get_collection(name)
            print(f"  ✓ Collection '{name}' already exists, skipping.")
            continue
        except (UnexpectedResponse, Exception):
            pass

        # Create collection
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=col["vector_size"],
                distance=Distance.COSINE,
            ),
        )
        print(f"  ✓ Created collection '{name}'.")

        # Create payload indexes
        for field in col["payload_indexes"]:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        print(f"    Indexed: {', '.join(col['payload_indexes'])}")

    print("Done.")


if __name__ == "__main__":
    main()
