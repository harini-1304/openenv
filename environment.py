import threading
import time
from typing import Dict, Any, Optional, List
from models import EpisodeState, EmailClassification, AgentAction, Category, Response

class EmailTriageEnvironment:
    """Thread-safe email triage environment"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._state = EpisodeState()
        self._current_task = None
        self._task_history = []
        
    def reset(self) -> Dict[str, Any]:
        """Reset environment with thread safety"""
        with self._lock:
            # Import here to avoid circular imports
            from tasks import get_all_tasks
            tasks = get_all_tasks()
            
            # Select random task
            import random
            self._current_task = random.choice(tasks)
            self._task_history.append(self._current_task.task_id)
            
            # Reset state
            self._state.reset()
            
            return {
                "observation": {
                    "email": self._current_task.email,
                    "task_id": self._current_task.task_id
                },
                "done": False,
                "reward": 0.0
            }
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Thread-safe step with validation"""
        with self._lock:
            if self._state.is_done:
                raise RuntimeError("Cannot step after episode is done. Call reset() first.")
            
            if not self._current_task:
                raise RuntimeError("No active task. Call reset() first.")
            
            # Validate action
            try:
                agent_action = AgentAction(**action)
            except Exception as e:
                return {
                    "observation": {},
                    "done": True,
                    "reward": 0.0,
                    "error": f"Invalid action format: {e}"
                }
            
            # Calculate reward based on ground truth
            reward = self._calculate_reward(agent_action)
            
            # Update state
            self._state.add_step(action, reward)
            
            # Check if episode should end (single-step episodes)
            self._state.is_done = True
            
            return {
                "observation": {
                    "email": self._current_task.email,
                    "task_id": self._current_task.task_id
                },
                "done": self._state.is_done,
                "reward": reward,
                "info": {
                    "ground_truth": {
                        "category": self._current_task.category,
                        "response": self._current_task.response,
                        "explanation": self._current_task.explanation
                    },
                    "step": self._state.current_step,
                    "total_reward": self._state.episode_reward
                }
            }
    
    def _calculate_reward(self, action: AgentAction) -> float:
        """Calculate reward based on ground truth"""
        if not self._current_task:
            return 0.0
        
        # Category correctness (50% weight)
        category_correct = 1.0 if action.category == self._current_task.category else 0.0
        
        # Response correctness (30% weight)
        response_correct = 1.0 if action.response == self._current_task.response else 0.0
        
        # Confidence bonus (10% weight)
        confidence_bonus = action.confidence if (category_correct and response_correct) else 0.0
        
        # Efficiency bonus (10% weight) - early steps get bonus
        efficiency_bonus = 0.1 if self._state.current_step <= 1 else 0.0
        
        # Total reward
        total_reward = (
            category_correct * 0.5 +
            response_correct * 0.3 +
            confidence_bonus * 0.1 +
            efficiency_bonus * 0.1
        )
        
        # Clamp to valid range
        return max(0.1, min(0.95, total_reward))
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state (thread-safe)"""
        with self._lock:
            return {
                "current_step": self._state.current_step,
                "total_steps": self._state.total_steps,
                "is_done": self._state.is_done,
                "episode_reward": self._state.episode_reward,
                "task_history": self._task_history.copy()
            }
    
    def get_current_task(self):
        """Get current task (thread-safe)"""
        with self._lock:
            return self._current_task
    
    def validate_constraints(self, action: Dict[str, Any]) -> bool:
        """Validate business constraints"""
        try:
            agent_action = AgentAction(**action)
            
            # Constraint: spam must be ignored
            if agent_action.category == Category.SPAM and agent_action.response != Response.IGNORE:
                return False
            
            # Constraint: escalate only with urgent
            if agent_action.response == Response.ESCALATE and agent_action.category != Category.URGENT:
                return False
            
            return True
        except Exception:
            return False
