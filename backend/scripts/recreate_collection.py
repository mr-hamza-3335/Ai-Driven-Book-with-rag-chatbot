"""Script to recreate Qdrant collection with proper indexes."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

# Load environment
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "book_chunks"
VECTOR_SIZE = 768  # Gemini text-embedding-004

def main():
    print("=" * 60)
    print("Recreating Qdrant Collection")
    print("=" * 60)

    # Connect to Qdrant
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(f"Connected to Qdrant: {QDRANT_URL}")

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME in collection_names:
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
        print("Collection deleted.")

    # Create new collection
    print(f"Creating collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print("Collection created.")

    # Create payload indexes
    print("Creating payload index for chapter_id...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="chapter_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("Index created.")

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"\nCollection info:")
    print(f"  Name: {COLLECTION_NAME}")
    print(f"  Points: {info.points_count}")
    print(f"  Status: {info.status}")

    print("\nCollection recreated successfully!")

if __name__ == "__main__":
    main()
