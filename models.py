from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class Category(str, Enum):
    URGENT = "urgent"
    NORMAL = "normal"
    SPAM = "spam"

class Response(str, Enum):
    REPLY = "reply"
    IGNORE = "ignore"
    ESCALATE = "escalate"

class Department(str, Enum):
    TECH = "tech"
    FINANCE = "finance"
    SALES = "sales"
    GENERAL = "general"

class EmailClassification(BaseModel):
    """Type-safe email classification with validation"""
    category: Category = Field(..., description="Email category")
    response: Response = Field(..., description="Recommended action")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reason: Optional[str] = Field(None, description="Classification reason")
    
    @field_validator('category', mode='before')
    @classmethod
    def normalize_category(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v
    
    @field_validator('response', mode='before')
    @classmethod
    def normalize_response(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v
    
    @field_validator('confidence', mode='before')
    @classmethod
    def validate_confidence(cls, v):
        if isinstance(v, (int, float)):
            return float(max(0.0, min(1.0, v)))
        return v

class AgentAction(BaseModel):
    """Agent action with business context"""
    category: Category
    response: Response
    priority: float = Field(..., ge=0.0, le=1.0)
    department: Department
    urgency_reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    @field_validator('priority', mode='before')
    @classmethod
    def validate_priority(cls, v):
        if isinstance(v, (int, float)):
            return float(max(0.0, min(1.0, v)))
        return v

class EpisodeState(BaseModel):
    """Thread-safe episode state"""
    current_step: int = Field(default=0, ge=0)
    total_steps: int = Field(default=9, ge=1, le=20)
    is_done: bool = Field(default=False)
    episode_reward: float = Field(default=0.0)
    history: List[Dict[str, Union[str, float]]] = Field(default_factory=list)
    
    def add_step(self, action: Dict[str, Union[str, float]], reward: float):
        """Add step to history"""
        self.history.append({**action, "reward": reward})
        self.episode_reward += reward
        self.current_step += 1
    
    def reset(self):
        """Reset episode state"""
        self.current_step = 0
        self.is_done = False
        self.episode_reward = 0.0
        self.history = []

class RuleSignal(BaseModel):
    """Individual rule signal for pattern matching"""
    signal_type: str  # "keyword", "pattern", "context"
    value: str
    weight: float = Field(ge=0.0, le=1.0)
    matches: bool = False

class RuleMatch(BaseModel):
    """Rule match with confidence calculation"""
    signals: List[RuleSignal]
    total_confidence: float = Field(ge=0.0, le=1.0)
    category: Category
    response: Response
    reason: str

class LLMResponse(BaseModel):
    """Validated LLM response"""
    category: Category
    response: Response
    confidence: float
    reasoning: Optional[str] = None
    
    @field_validator('category', 'response', mode='before')
    @classmethod
    def normalize_fields(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v
