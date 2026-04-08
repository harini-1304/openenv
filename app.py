from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# Complete 9-task email triage environment
EMAIL_TASKS = [
    {"id": 0, "email": "Please reset my password. I forgot it.", "category": "normal", "response": "reply"},
    {"id": 1, "email": "Claim your free prize now! Click here!", "category": "spam", "response": "ignore"},
    {"id": 2, "email": "What are your business hours?", "category": "normal", "response": "reply"},
    {"id": 3, "email": "The server is running slow today.", "category": "urgent", "response": "escalate"},
    {"id": 4, "email": "Meeting reminder: Tomorrow at 2 PM.", "category": "urgent", "response": "reply"},
    {"id": 5, "email": "Special offer just for you!", "category": "spam", "response": "ignore"},
    {"id": 6, "email": "CRITICAL: Database corruption detected!", "category": "urgent", "response": "escalate"},
    {"id": 7, "email": "Your account will be suspended unless you verify.", "category": "spam", "response": "ignore"},
    {"id": 8, "email": "Q3 financial report shows concerning trends.", "category": "urgent", "response": "escalate"}
]

class Action(BaseModel):
    category: str
    response: str

class EmailTriageEnv:
    def __init__(self):
        self.current_task_index = 0
        self.tasks = EMAIL_TASKS
        self.episode_done = False
        self.total_reward = 0.0
        
    def reset(self) -> Dict[str, Any]:
        self.current_task_index = 0
        self.episode_done = False
        self.total_reward = 0.0
        
        if self.tasks:
            task = self.tasks[0]
            return {
                "observation": {
                    "email": task["email"],
                    "task_id": task["id"]
                },
                "reward": 0.0,
                "done": False,
                "info": {}
            }
        return {"observation": None, "reward": 0.0, "done": True, "info": {}}
    
    def step(self, action: Action) -> Dict[str, Any]:
        if self.episode_done or self.current_task_index >= len(self.tasks):
            return {"observation": None, "reward": 0.0, "done": True, "info": {}}
        
        current_task = self.tasks[self.current_task_index]
        
        category_correct = action.category.lower() == current_task["category"].lower()
        response_correct = action.response.lower() == current_task["response"].lower()
        
        reward = 0.0
        if category_correct:
            reward += 0.5
        if response_correct:
            reward += 0.5
            
        self.total_reward += reward
        self.current_task_index += 1
        
        if self.current_task_index >= len(self.tasks):
            self.episode_done = True
            return {
                "observation": None,
                "reward": reward,
                "done": True,
                "info": {
                    "total_score": self.total_reward,
                    "category_correct": category_correct,
                    "response_correct": response_correct
                }
            }
        
        next_task = self.tasks[self.current_task_index]
        
        return {
            "observation": {
                "email": next_task["email"],
                "task_id": next_task["id"]
            },
            "reward": reward,
            "done": False,
            "info": {
                "category_correct": category_correct,
                "response_correct": response_correct,
                "total_score": self.total_reward
            }
        }
    
    def get_state(self) -> Dict[str, Any]:
        if self.current_task_index >= len(self.tasks):
            return {
                "current_task": None,
                "task_index": self.current_task_index,
                "total_tasks": len(self.tasks),
                "episode_done": True,
                "total_reward": self.total_reward
            }
        
        current_task = self.tasks[self.current_task_index]
        return {
            "current_task": {
                "email": current_task["email"],
                "task_id": current_task["id"]
            },
            "task_index": self.current_task_index,
            "total_tasks": len(self.tasks),
            "episode_done": self.episode_done,
            "total_reward": self.total_reward
        }

app = FastAPI(title="Email Triage Environment", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmailTriageEnv()

@app.get("/")
def root():
    return {
        "status": "running",
        "environment": "email_triage",
        "version": "1.0.0",
        "total_tasks": len(env.tasks)
    }

@app.post("/reset")
def reset():
    result = env.reset()
    return result

@app.post("/step")
def step(action: Action):
    result = env.step(action)
    return result

@app.get("/state")
def state():
    result = env.get_state()
    return result

@app.options("/reset")
@app.options("/step")
@app.options("/state")
def handle_options():
    return {"message": "OK"}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
