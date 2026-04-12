from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class EmailTask(BaseModel):
    """Email triage task with ground truth and explanations"""
    email: str = Field(..., description="The email content to classify")
    task_id: int = Field(..., description="Unique task identifier")
    
    # Ground truth with explanations
    category: str = Field(..., description="Ground truth category: urgent|normal|spam")
    response: str = Field(..., description="Ground truth response: reply|ignore|escalate")
    explanation: str = Field(..., description="Why this label is correct")
    
    # Constraints for validation
    @classmethod
    def validate_constraints(cls, category: str, response: str) -> bool:
        """Enforce business constraints"""
        if category == "spam" and response != "ignore":
            return False  # Spam must always be ignored
        if response == "escalate" and category != "urgent":
            return False  # Escalate only with urgent
        return True

# Predefined tasks with explanations for auditability
EMAIL_TASKS = [
    EmailTask(
        email="URGENT: Server database is down - immediate attention required!",
        task_id=1,
        category="urgent",
        response="escalate",
        explanation="Technical emergency with 'database down' keywords requires immediate escalation"
    ),
    EmailTask(
        email="Meeting invitation for next Tuesday at 2pm",
        task_id=2,
        category="normal",
        response="reply",
        explanation="Standard business meeting requires normal response"
    ),
    EmailTask(
        email="FREE PRIZE! You've won $1,000,000! Claim now!",
        task_id=3,
        category="spam",
        response="ignore",
        explanation="Classic spam indicators: 'FREE PRIZE', exclamation marks, urgency without context"
    ),
    EmailTask(
        email="Re: Project deadline approaching - need status update",
        task_id=4,
        category="urgent",
        response="reply",
        explanation="Reply thread with deadline indicates urgency but response is appropriate"
    ),
    EmailTask(
        email="Invoice #12345 - Payment due in 30 days",
        task_id=5,
        category="normal",
        response="reply",
        explanation="Financial document requires standard business response"
    ),
    EmailTask(
        email="CRITICAL: System infrastructure failure - all services down",
        task_id=6,
        category="urgent",
        response="escalate",
        explanation="Infrastructure failure with multiple critical keywords requires escalation"
    ),
    EmailTask(
        email="Limited time offer - 50% discount on everything!",
        task_id=7,
        category="spam",
        response="ignore",
        explanation="Marketing spam with 'limited time offer' pattern"
    ),
    EmailTask(
        email="Re: Customer complaint - urgent resolution needed",
        task_id=8,
        category="urgent",
        response="reply",
        explanation="Customer complaint in reply thread needs urgent attention but reply is appropriate"
    ),
    EmailTask(
        email="Weekly team sync meeting agenda",
        task_id=9,
        category="normal",
        response="reply",
        explanation="Regular business communication requires normal response"
    )
]

def get_task_by_id(task_id: int) -> Optional[EmailTask]:
    """Get task by ID with validation"""
    for task in EMAIL_TASKS:
        if task.task_id == task_id:
            # Validate constraints
            if not EmailTask.validate_constraints(task.category, task.response):
                raise ValueError(f"Task {task_id} violates constraints: {task.category}/{task.response}")
            return task
    return None

def get_all_tasks() -> List[EmailTask]:
    """Get all validated tasks"""
    validated_tasks = []
    for task in EMAIL_TASKS:
        if EmailTask.validate_constraints(task.category, task.response):
            validated_tasks.append(task)
        else:
            raise ValueError(f"Task {task.task_id} violates constraints: {task.category}/{task.response}")
    return validated_tasks
