"""MongoDB-backed deduplication store."""

from datetime import datetime, timezone
from typing import ClassVar

from motor.motor_asyncio import AsyncIOMotorClient

from jobapply.settings import get_settings
from jobapply.utils.redaction import redact_data


class DeduplicationStore:
    """Persistent tracking of seen job IDs across sessions."""

    _client: ClassVar[AsyncIOMotorClient | None] = None

    def __init__(self):
        """Initialize deduplication store with MongoDB connection."""
        settings = get_settings()
        if self.__class__._client is None:
            self.__class__._client = AsyncIOMotorClient(settings.mongodb_url)
        self.client = self.__class__._client
        self.collection = self.client[settings.mongodb_db][settings.seen_jobs_collection]

    async def is_seen(self, job_id: str) -> bool:
        """Check if a job ID has been seen before.

        Args:
            job_id: Exact job ID to check.

        Returns:
            True if job has been seen, False otherwise.
        """
        return await self.collection.find_one({"job_id": job_id}) is not None

    async def mark_seen(self, job_id: str, metadata: dict) -> None:
        """Mark a job as seen with metadata.

        Args:
            job_id: Exact job ID to mark as seen.
            metadata: Additional metadata to store (title, company, url, qualification data, etc.).
        """
        safe_metadata = redact_data(metadata)
        await self.collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "job_id": job_id,
                    **safe_metadata,
                    "evaluated_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    async def load_seen_ids(self, for_live_run: bool = False) -> set[str]:
        """Load all seen job IDs into memory for fast lookup.

        Returns:
            Set of all seen job IDs.
        """
        query = {"status": {"$nin": ["qualified", "dry_run"]}} if for_live_run else {}
        cursor = self.collection.find(query, {"job_id": 1})
        return {doc["job_id"] async for doc in cursor}

    async def get_daily_count(self) -> int:
        """Count applications submitted today.

        Returns:
            Number of applications submitted today.
        """
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return await self.collection.count_documents(
            {
                "applied_at": {"$gte": today},
                "status": "submitted",
            }
        )

    @classmethod
    async def close(cls) -> None:
        """Close the shared MongoDB client at application shutdown."""
        if cls._client is not None:
            cls._client.close()
            cls._client = None
