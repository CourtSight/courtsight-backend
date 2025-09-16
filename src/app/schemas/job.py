from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class JobBase(BaseModel):
    """Base job schema."""
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class Job(JobBase):
    """Job read schema."""
    id: str
    created_at: datetime
    updated_at: datetime


class JobCreate(JobBase):
    """Job creation schema."""
    pass


class JobRead(Job):
    """Job read schema (alias for Job)."""
    pass


class JobUpdate(BaseModel):
    """Job update schema."""
    status: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
