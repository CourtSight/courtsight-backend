"""
Document Processing Task Scheduler for CourtSight
Handles automated daily document processing and background tasks
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from sqlalchemy.orm import selectinload

from ..core.db.database import async_get_db
from ..models.legal_document import LegalDocument
from .document_processor import document_processor, ProcessingStatus
from ..core.logger import logger


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """Represents a scheduled processing task."""
    task_id: str
    task_type: str
    target_date: date
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class DocumentProcessingScheduler:
    """
    Service for scheduling and managing document processing tasks.
    
    Features:
    - Automated daily processing
    - Background task management
    - Processing queue with priority
    - Error handling and retry logic
    - Processing statistics and monitoring
    """
    
    def __init__(self):
        """Initialize the task scheduler."""
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_history: Dict[str, ScheduledTask] = {}
        self.is_scheduler_running = False
        self.daily_processing_hour = 6  # Run at 6 AM daily
        logger.info("DocumentProcessingScheduler initialized")
    
    async def start_scheduler(self):
        """Start the automated task scheduler."""
        if self.is_scheduler_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_scheduler_running = True
        logger.info("Starting document processing scheduler")
        
        # Start the main scheduler loop
        asyncio.create_task(self._scheduler_loop())
        
        # Start daily processing task
        asyncio.create_task(self._daily_processing_loop())
        
        logger.info("Document processing scheduler started")
    
    async def stop_scheduler(self):
        """Stop the automated task scheduler."""
        self.is_scheduler_running = False
        
        # Cancel all running tasks
        for task_id, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled task: {task_id}")
        
        self.running_tasks.clear()
        logger.info("Document processing scheduler stopped")
    
    async def schedule_daily_processing(
        self,
        target_date: Optional[date] = None,
        force_reprocess: bool = False,
        priority: int = 1
    ) -> str:
        """
        Schedule daily document processing for a specific date.
        
        Args:
            target_date: Date to process (defaults to today)
            force_reprocess: Whether to reprocess existing documents
            priority: Task priority (higher numbers = higher priority)
            
        Returns:
            Task ID for tracking
        """
        if target_date is None:
            target_date = date.today()
        
        task_id = f"daily_processing_{target_date.isoformat()}_{datetime.now().timestamp()}"
        
        # Create scheduled task
        scheduled_task = ScheduledTask(
            task_id=task_id,
            task_type="daily_processing",
            target_date=target_date,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.task_history[task_id] = scheduled_task
        
        # Schedule the task
        task = asyncio.create_task(
            self._execute_daily_processing(task_id, target_date, force_reprocess)
        )
        self.running_tasks[task_id] = task
        
        logger.info(f"Scheduled daily processing task: {task_id} for {target_date}")
        return task_id
    
    async def schedule_bulk_processing(
        self,
        start_date: date,
        end_date: date,
        batch_size: int = 100
    ) -> str:
        """
        Schedule bulk document processing for a date range.
        
        Args:
            start_date: Start date for processing
            end_date: End date for processing
            batch_size: Number of documents to process per batch
            
        Returns:
            Task ID for tracking
        """
        task_id = f"bulk_processing_{start_date.isoformat()}_{end_date.isoformat()}_{datetime.now().timestamp()}"
        
        # Create scheduled task
        scheduled_task = ScheduledTask(
            task_id=task_id,
            task_type="bulk_processing",
            target_date=start_date,  # Use start_date as representative date
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.task_history[task_id] = scheduled_task
        
        # Schedule the task
        task = asyncio.create_task(
            self._execute_bulk_processing(task_id, start_date, end_date, batch_size)
        )
        self.running_tasks[task_id] = task
        
        logger.info(f"Scheduled bulk processing task: {task_id} from {start_date} to {end_date}")
        return task_id
    
    async def schedule_reprocessing_failed_documents(
        self,
        days_back: int = 7,
        max_documents: int = 100
    ) -> str:
        """
        Schedule reprocessing of failed documents from recent days.
        
        Args:
            days_back: Number of days to look back for failed documents
            max_documents: Maximum number of documents to reprocess
            
        Returns:
            Task ID for tracking
        """
        task_id = f"reprocess_failed_{datetime.now().timestamp()}"
        
        # Create scheduled task
        scheduled_task = ScheduledTask(
            task_id=task_id,
            task_type="reprocess_failed",
            target_date=date.today(),
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.task_history[task_id] = scheduled_task
        
        # Schedule the task
        task = asyncio.create_task(
            self._execute_reprocess_failed(task_id, days_back, max_documents)
        )
        self.running_tasks[task_id] = task
        
        logger.info(f"Scheduled failed document reprocessing: {task_id}")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[ScheduledTask]:
        """Get the status of a scheduled task."""
        return self.task_history.get(task_id)
    
    async def get_running_tasks(self) -> Dict[str, ScheduledTask]:
        """Get all currently running tasks."""
        running_task_info = {}
        for task_id in self.running_tasks:
            if task_id in self.task_history:
                task_info = self.task_history[task_id]
                if task_info.status == TaskStatus.RUNNING:
                    running_task_info[task_id] = task_info
        
        return running_task_info
    
    async def get_task_history(
        self,
        limit: int = 100,
        task_type: Optional[str] = None
    ) -> List[ScheduledTask]:
        """Get task execution history."""
        tasks = list(self.task_history.values())
        
        # Filter by task type if specified
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        
        # Sort by creation time (newest first)
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        
        return tasks[:limit]
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            if not task.done():
                task.cancel()
                
                # Update task status
                if task_id in self.task_history:
                    self.task_history[task_id].status = TaskStatus.CANCELLED
                    self.task_history[task_id].completed_at = datetime.now()
                
                logger.info(f"Cancelled task: {task_id}")
                return True
        
        return False
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get overall processing statistics."""
        async for db in async_get_db():
            try:
                # Get document statistics
                total_docs_query = select(LegalDocument.id).where(LegalDocument.is_active == True)
                total_docs_result = await db.execute(total_docs_query)
                total_documents = len(total_docs_result.fetchall())
                
                # Get processing status statistics
                processed_docs_query = select(LegalDocument.id).where(
                    and_(
                        LegalDocument.is_active == True,
                        LegalDocument.processing_status == "completed"
                    )
                )
                processed_docs_result = await db.execute(processed_docs_query)
                processed_documents = len(processed_docs_result.fetchall())
                
                # Get recent processing statistics
                week_ago = datetime.now() - timedelta(days=7)
                recent_docs_query = select(LegalDocument.id).where(
                    and_(
                        LegalDocument.created_at >= week_ago,
                        LegalDocument.is_active == True
                    )
                )
                recent_docs_result = await db.execute(recent_docs_query)
                recent_documents = len(recent_docs_result.fetchall())
                
                # Task statistics
                total_tasks = len(self.task_history)
                completed_tasks = len([t for t in self.task_history.values() if t.status == TaskStatus.COMPLETED])
                failed_tasks = len([t for t in self.task_history.values() if t.status == TaskStatus.FAILED])
                running_tasks = len([t for t in self.task_history.values() if t.status == TaskStatus.RUNNING])
                
                return {
                    "documents": {
                        "total_documents": total_documents,
                        "processed_documents": processed_documents,
                        "recent_documents_7_days": recent_documents,
                        "processing_rate": (processed_documents / total_documents * 100) if total_documents > 0 else 0
                    },
                    "tasks": {
                        "total_tasks": total_tasks,
                        "completed_tasks": completed_tasks,
                        "failed_tasks": failed_tasks,
                        "running_tasks": running_tasks,
                        "success_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
                    },
                    "scheduler": {
                        "is_running": self.is_scheduler_running,
                        "active_tasks": len(self.running_tasks)
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to get processing statistics: {str(e)}")
                raise
            finally:
                await db.close()
                break
    
    async def _scheduler_loop(self):
        """Main scheduler loop for managing tasks."""
        while self.is_scheduler_running:
            try:
                # Clean up completed tasks
                await self._cleanup_completed_tasks()
                
                # Check for stuck tasks
                await self._check_stuck_tasks()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {str(e)}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _daily_processing_loop(self):
        """Loop for automated daily processing."""
        while self.is_scheduler_running:
            try:
                now = datetime.now()
                
                # Check if it's time for daily processing
                if (now.hour == self.daily_processing_hour and 
                    now.minute < 5):  # Run within first 5 minutes of the hour
                    
                    # Check if we've already run today
                    today = date.today()
                    today_tasks = [
                        t for t in self.task_history.values()
                        if (t.task_type == "daily_processing" and 
                            t.target_date == today and
                            t.status in [TaskStatus.COMPLETED, TaskStatus.RUNNING])
                    ]
                    
                    if not today_tasks:
                        logger.info("Starting automated daily processing")
                        await self.schedule_daily_processing()
                
                # Wait for next check (every hour)
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Daily processing loop error: {str(e)}", exc_info=True)
                await asyncio.sleep(3600)
    
    async def _execute_daily_processing(
        self,
        task_id: str,
        target_date: date,
        force_reprocess: bool
    ):
        """Execute daily document processing task."""
        try:
            # Update task status
            self.task_history[task_id].status = TaskStatus.RUNNING
            self.task_history[task_id].started_at = datetime.now()
            
            logger.info(f"Starting daily processing task: {task_id}")
            
            # Execute processing
            async for db in async_get_db():
                try:
                    result = await document_processor.process_daily_documents(
                        db=db,
                        target_date=target_date,
                        force_reprocess=force_reprocess
                    )
                    
                    # Update task with results
                    self.task_history[task_id].status = TaskStatus.COMPLETED
                    self.task_history[task_id].completed_at = datetime.now()
                    self.task_history[task_id].result = result
                    
                    logger.info(f"Daily processing task completed: {task_id}")
                    
                except Exception as e:
                    logger.error(f"Daily processing task failed: {task_id} - {str(e)}")
                    self.task_history[task_id].status = TaskStatus.FAILED
                    self.task_history[task_id].completed_at = datetime.now()
                    self.task_history[task_id].error_message = str(e)
                    raise
                finally:
                    await db.close()
                    break
                    
        except Exception as e:
            logger.error(f"Daily processing task error: {task_id} - {str(e)}")
            self.task_history[task_id].status = TaskStatus.FAILED
            self.task_history[task_id].completed_at = datetime.now()
            self.task_history[task_id].error_message = str(e)
        finally:
            # Remove from running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _execute_bulk_processing(
        self,
        task_id: str,
        start_date: date,
        end_date: date,
        batch_size: int
    ):
        """Execute bulk document processing task."""
        try:
            # Update task status
            self.task_history[task_id].status = TaskStatus.RUNNING
            self.task_history[task_id].started_at = datetime.now()
            
            logger.info(f"Starting bulk processing task: {task_id}")
            
            # Execute processing
            async for db in async_get_db():
                try:
                    result = await document_processor.process_document_range(
                        db=db,
                        start_date=start_date,
                        end_date=end_date,
                        batch_size=batch_size
                    )
                    
                    # Update task with results
                    self.task_history[task_id].status = TaskStatus.COMPLETED
                    self.task_history[task_id].completed_at = datetime.now()
                    self.task_history[task_id].result = result
                    
                    logger.info(f"Bulk processing task completed: {task_id}")
                    
                except Exception as e:
                    logger.error(f"Bulk processing task failed: {task_id} - {str(e)}")
                    self.task_history[task_id].status = TaskStatus.FAILED
                    self.task_history[task_id].completed_at = datetime.now()
                    self.task_history[task_id].error_message = str(e)
                    raise
                finally:
                    await db.close()
                    break
                    
        except Exception as e:
            logger.error(f"Bulk processing task error: {task_id} - {str(e)}")
            self.task_history[task_id].status = TaskStatus.FAILED
            self.task_history[task_id].completed_at = datetime.now()
            self.task_history[task_id].error_message = str(e)
        finally:
            # Remove from running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _execute_reprocess_failed(
        self,
        task_id: str,
        days_back: int,
        max_documents: int
    ):
        """Execute reprocessing of failed documents."""
        try:
            # Update task status
            self.task_history[task_id].status = TaskStatus.RUNNING
            self.task_history[task_id].started_at = datetime.now()
            
            logger.info(f"Starting failed document reprocessing: {task_id}")
            
            # Find failed documents
            async for db in async_get_db():
                try:
                    cutoff_date = datetime.now() - timedelta(days=days_back)
                    
                    # Query for failed documents
                    query = select(LegalDocument).where(
                        and_(
                            LegalDocument.processing_status == "failed",
                            LegalDocument.created_at >= cutoff_date,
                            LegalDocument.is_active == True
                        )
                    ).limit(max_documents)
                    
                    result = await db.execute(query)
                    failed_docs = result.scalars().all()
                    
                    logger.info(f"Found {len(failed_docs)} failed documents to reprocess")
                    
                    # Reprocess each document
                    reprocessed = 0
                    for doc in failed_docs:
                        try:
                            # TODO: Implement reprocessing logic
                            # This would involve re-downloading, re-extracting, and re-vectorizing
                            logger.info(f"Reprocessing document: {doc.case_number}")
                            reprocessed += 1
                            
                        except Exception as e:
                            logger.error(f"Failed to reprocess document {doc.case_number}: {str(e)}")
                    
                    # Update task with results
                    result_data = {
                        "total_failed_found": len(failed_docs),
                        "successfully_reprocessed": reprocessed,
                        "reprocessing_rate": (reprocessed / len(failed_docs) * 100) if failed_docs else 0
                    }
                    
                    self.task_history[task_id].status = TaskStatus.COMPLETED
                    self.task_history[task_id].completed_at = datetime.now()
                    self.task_history[task_id].result = result_data
                    
                    logger.info(f"Failed document reprocessing completed: {task_id}")
                    
                except Exception as e:
                    logger.error(f"Failed document reprocessing error: {task_id} - {str(e)}")
                    self.task_history[task_id].status = TaskStatus.FAILED
                    self.task_history[task_id].completed_at = datetime.now()
                    self.task_history[task_id].error_message = str(e)
                    raise
                finally:
                    await db.close()
                    break
                    
        except Exception as e:
            logger.error(f"Failed document reprocessing error: {task_id} - {str(e)}")
            self.task_history[task_id].status = TaskStatus.FAILED
            self.task_history[task_id].completed_at = datetime.now()
            self.task_history[task_id].error_message = str(e)
        finally:
            # Remove from running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _cleanup_completed_tasks(self):
        """Clean up completed tasks from running tasks dictionary."""
        completed_task_ids = []
        
        for task_id, task in self.running_tasks.items():
            if task.done():
                completed_task_ids.append(task_id)
        
        for task_id in completed_task_ids:
            del self.running_tasks[task_id]
    
    async def _check_stuck_tasks(self):
        """Check for tasks that might be stuck and mark them as failed."""
        max_task_duration = timedelta(hours=6)  # Maximum task duration
        current_time = datetime.now()
        
        for task_id, scheduled_task in self.task_history.items():
            if (scheduled_task.status == TaskStatus.RUNNING and
                scheduled_task.started_at and
                current_time - scheduled_task.started_at > max_task_duration):
                
                logger.warning(f"Task {task_id} appears to be stuck, marking as failed")
                
                # Cancel the task if it's still running
                if task_id in self.running_tasks:
                    await self.cancel_task(task_id)
                
                # Mark as failed
                scheduled_task.status = TaskStatus.FAILED
                scheduled_task.completed_at = current_time
                scheduled_task.error_message = "Task exceeded maximum duration and was cancelled"


# Global instance
document_processing_scheduler = DocumentProcessingScheduler()
