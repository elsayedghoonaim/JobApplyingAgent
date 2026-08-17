"""Data models for Telegram correlations, update cursors, and notification outbox."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class OutboxStatus(str, Enum):
    """Status of an outbox notification record."""

    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    DELIVERY_UNKNOWN = "delivery_unknown"
    FAILED = "failed"


class CorrelationStatus(str, Enum):
    """Lifecycle status for a Telegram human-in-the-loop correlation."""

    PENDING = "pending"
    REPLIED = "replied"
    CONSUMED = "consumed"
    TIMED_OUT = "timed_out"
    PROMPT_DELIVERY_UNKNOWN = "prompt_delivery_unknown"
    PROMPT_FAILED = "prompt_failed"
    STORAGE_ERROR = "storage_error"


class OutboxChunk(BaseModel):
    """One chunk of a multi-chunk notification payload."""

    chunk_index: int
    text: str
    sent: bool = False
    in_flight: bool = False
    message_id: Optional[int] = None
    sent_at: Optional[datetime] = None


class OutboxRecord(BaseModel):
    """Durable notification outbox record."""

    idempotency_key: str
    notification_type: str
    run_id: str
    job_id: Optional[str] = None
    payload_hash: str
    chunks: list[OutboxChunk] = Field(default_factory=list)
    status: OutboxStatus = OutboxStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    next_attempt_at: Optional[datetime] = None
    lease_id: Optional[str] = None
    lease_expires_at: Optional[datetime] = None
    error_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    sent_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutboxEnqueueResult(BaseModel):
    """Result of enqueuing a notification to the outbox."""

    enqueued: bool
    already_exists: bool = False
    status: OutboxStatus
    idempotency_key: str
    reason: Optional[str] = None


class OutboxClaimResult(BaseModel):
    """Result of claiming due outbox records under a lease."""

    claimed: bool
    records: list[OutboxRecord] = Field(default_factory=list)
    reason: Optional[str] = None


class OutboxDeliveryResult(BaseModel):
    """Result of attempting delivery for an outbox record."""

    success: bool
    status: OutboxStatus
    idempotency_key: str
    chunks_sent: int = 0
    total_chunks: int = 0
    reason: Optional[str] = None


class OutboxDrainResult(BaseModel):
    """Result of draining due outbox records."""

    total_processed: int = 0
    sent_count: int = 0
    failed_count: int = 0
    unknown_count: int = 0
    pending_count: int = 0
    errors: list[str] = Field(default_factory=list)


class TelegramCorrelation(BaseModel):
    """Durable Telegram correlation for approval or form questions."""

    correlation_key: str
    purpose: str
    run_id: str
    job_id: Optional[str] = None
    prompt_hash: str
    nonce: str
    prompt_text: str
    prompt_in_flight: bool = False
    prompt_message_id: Optional[int] = None
    prompt_sent_at: Optional[datetime] = None
    status: CorrelationStatus = CorrelationStatus.PENDING
    lease_id: Optional[str] = None
    lease_expires_at: Optional[datetime] = None
    reply_text: Optional[str] = None
    reply_update_id: Optional[int] = None
    reply_message_id: Optional[int] = None
    replied_at: Optional[datetime] = None
    consumed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TelegramCursor(BaseModel):
    """Durable Telegram getUpdates polling cursor."""

    bot_chat_key: str
    last_update_id: int
    lease_id: Optional[str] = None
    lease_expires_at: Optional[datetime] = None
    updated_at: datetime


class CorrelationRegisterResult(BaseModel):
    """Result of registering a correlation intent."""

    registered: bool
    is_new: bool
    correlation_key: str
    nonce: str
    prompt_text: str
    prompt_in_flight: bool = False
    prompt_message_id: Optional[int] = None
    status: CorrelationStatus
    reply_text: Optional[str] = None
    reason: Optional[str] = None


class CorrelationLeaseResult(BaseModel):
    """Result of claiming a correlation waiter lease."""

    acquired: bool
    lease_id: Optional[str] = None
    status: CorrelationStatus = CorrelationStatus.PENDING
    reply_text: Optional[str] = None
    reason: Optional[str] = None


class CorrelationReplyResult(BaseModel):
    """Result of saving a received correlated reply."""

    accepted: bool
    status: CorrelationStatus
    reply_text: Optional[str] = None
    reason: Optional[str] = None


class CorrelationConsumeResult(BaseModel):
    """Result of consuming a correlated reply."""

    consumed: bool
    reply_text: Optional[str] = None
    reason: Optional[str] = None


class CorrelationWaitResult(BaseModel):
    """Result of prompt delivery and correlated waiting."""

    status: CorrelationStatus
    reply_text: Optional[str] = None
    timed_out: bool = False
    nonce: Optional[str] = None
    error_reason: Optional[str] = None
