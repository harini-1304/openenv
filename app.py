import gradio as gr
import json
import random
from typing import Dict, Any, Tuple

# Email triage tasks with varying difficulty
EMAIL_TASKS = [
    # Easy tasks
    {
        "id": 0,
        "email": "Please reset my password. I forgot it.",
        "sender": "user@example.com",
        "subject": "Password Reset Request",
        "category": "normal",
        "response": "reply",
        "difficulty": "easy"
    },
    {
        "id": 1,
        "email": "Claim your free prize now! Click here!",
        "sender": "spam@suspicious.com",
        "subject": "You Won!!!",
        "category": "spam",
        "response": "ignore",
        "difficulty": "easy"
    },
    {
        "id": 2,
        "email": "What are your business hours?",
        "sender": "customer@company.com",
        "subject": "Question",
        "category": "normal",
        "response": "reply",
        "difficulty": "easy"
    },
    
    # Medium tasks
    {
        "id": 3,
        "email": "The server is running slow today. Can someone check?",
        "sender": "employee@company.com",
        "subject": "Performance Issue",
        "category": "urgent",
        "response": "escalate",
        "difficulty": "medium"
    },
    {
        "id": 4,
        "email": "Meeting reminder: Tomorrow at 2 PM. Also, the client reports issues.",
        "sender": "manager@company.com",
        "subject": "Meeting + Client Issues",
        "category": "urgent",
        "response": "reply",
        "difficulty": "medium"
    },
    {
        "id": 5,
        "email": "Special offer just for you! Limited time discount on our products.",
        "sender": "marketing@deals.com",
        "subject": "Exclusive Offer",
        "category": "spam",
        "response": "ignore",
        "difficulty": "medium"
    },
    
    # Hard tasks
    {
        "id": 6,
        "email": "CRITICAL: Database corruption detected. Multiple services down. Emergency maintenance required.",
        "sender": "sysadmin@company.com",
        "subject": "SYSTEM FAILURE - IMMEDIATE ACTION REQUIRED",
        "category": "urgent",
        "response": "escalate",
        "difficulty": "hard"
    },
    {
        "id": 7,
        "email": "Your account will be suspended unless you verify immediately. Click here: http://fake-bank.com",
        "sender": "security@bank-update.com",
        "subject": "URGENT: Account Verification Required",
        "category": "spam",
        "response": "ignore",
        "difficulty": "hard"
    },
    {
        "id": 8,
        "email": "Q3 financial report shows concerning trends. We need to discuss strategy and also the server needs patching.",
        "sender": "ceo@company.com",
        "subject": "Q3 Report + IT Infrastructure",
        "category": "urgent",
        "response": "escalate",
        "difficulty": "hard"
    }
]

class EmailTriageEnv:
    def __init__(self):
        self.current_task_index = 0
        self.tasks = EMAIL_TASKS
        self.episode_done = False
        self.total_reward = 0.0
        
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return first observation"""
        self.current_task_index = 0
        self.episode_done = False
        self.total_reward = 0.0
        
        if self.tasks:
            task = self.tasks[0]
            return {
                "observation": {
                    "email": task["email"],
                    "sender": task["sender"],
                    "subject": task["subject"],
                    "task_id": task["id"]
                },
                "reward": 0.0,
                "done": False,
                "info": {"difficulty": task["difficulty"]}
            }
        return {"observation": None, "reward": 0.0, "done": True, "info": {}}
    
    def step(self, action: Dict[str, str]) -> Dict[str, Any]:
        """Process action and return next observation and reward"""
        if self.episode_done or self.current_task_index >= len(self.tasks):
            return {"observation": None, "reward": 0.0, "done": True, "info": {}}
        
        current_task = self.tasks[self.current_task_index]
        
        # Calculate reward
        category_correct = action.get("category", "").lower() == current_task["category"].lower()
        response_correct = action.get("response", "").lower() == current_task["response"].lower()
        
        reward = 0.0
        if category_correct:
            reward += 0.5
        if response_correct:
            reward += 0.5
            
        self.total_reward += reward
        
        # Move to next task
        self.current_task_index += 1
        
        # Check if episode is done
        if self.current_task_index >= len(self.tasks):
            self.episode_done = True
            return {
                "observation": None,
                "reward": reward,
                "done": True,
                "info": {
                    "category_correct": category_correct,
                    "response_correct": response_correct,
                    "total_score": self.total_reward,
                    "episode_complete": True
                }
            }
        
        # Get next task
        next_task = self.tasks[self.current_task_index]
        
        return {
            "observation": {
                "email": next_task["email"],
                "sender": next_task["sender"],
                "subject": next_task["subject"],
                "task_id": next_task["id"]
            },
            "reward": reward,
            "done": False,
            "info": {
                "category_correct": category_correct,
                "response_correct": response_correct,
                "total_score": self.total_reward,
                "difficulty": next_task["difficulty"]
            }
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state"""
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
                "sender": current_task["sender"],
                "subject": current_task["subject"],
                "task_id": current_task["id"],
                "difficulty": current_task["difficulty"]
            },
            "task_index": self.current_task_index,
            "total_tasks": len(self.tasks),
            "episode_done": self.episode_done,
            "total_reward": self.total_reward
        }

# Global environment instance
env = EmailTriageEnv()

# Gradio interface functions
def reset_environment():
    """Reset the environment"""
    result = env.reset()
    if result["observation"]:
        return (
            f"**Email:** {result['observation']['email']}\n\n"
            f"**From:** {result['observation']['sender']}\n"
            f"**Subject:** {result['observation']['subject']}\n\n"
            f"**Task ID:** {result['observation']['task_id']}\n"
            f"**Difficulty:** {result['info']['difficulty']}\n\n"
            f"**Total Reward:** {result['reward']:.1f}",
            result['reward'],
            result['done'],
            f"Environment reset. Current task: {result['observation']['task_id']}"
        )
    return "No tasks available", 0.0, True, "Environment reset failed"

def process_action(category, response):
    """Process user action and return results"""
    if env.episode_done:
        return "Episode complete! Please reset the environment.", 0.0, True, "Episode already complete"
    
    action = {"category": category, "response": response}
    result = env.step(action)
    
    if result["done"]:
        message = (
            f"**Episode Complete!**\n\n"
            f"**Final Reward:** {result['reward']:.1f}\n"
            f"**Total Score:** {result['info']['total_score']:.1f}/9.0\n\n"
            f"**Performance:** {(result['info']['total_score']/9.0)*100:.1f}%\n\n"
            f"Congratulations! You completed all {len(env.tasks)} tasks!"
        )
        return message, result['reward'], True, "Episode completed successfully!"
    
    next_obs = result["observation"]
    message = (
        f"**Action Result:**\n\n"
        f"**Category Correct:** {'✅' if result['info']['category_correct'] else '❌'}\n"
        f"**Response Correct:** {'✅' if result['info']['response_correct'] else '❌'}\n"
        f"**Reward:** {result['reward']:.1f}\n"
        f"**Total Score:** {result['info']['total_score']:.1f}\n\n"
        f"**Next Email:**\n"
        f"**From:** {next_obs['sender']}\n"
        f"**Subject:** {next_obs['subject']}\n"
        f"**Email:** {next_obs['email']}\n\n"
        f"**Task ID:** {next_obs['task_id']}\n"
        f"**Difficulty:** {result['info']['difficulty']}"
    )
    
    return message, result['reward'], False, f"Action processed. Reward: {result['reward']:.1f}"

def get_environment_info():
    """Get current environment information"""
    state = env.get_state()
    if state["episode_done"]:
        return (
            f"**Environment Status:** Episode Complete\n\n"
            f"**Tasks Completed:** {state['task_index']}/{state['total_tasks']}\n"
            f"**Final Score:** {state['total_reward']:.1f}/9.0\n\n"
            f"**Performance:** {(state['total_reward']/9.0)*100:.1f}%"
        )
    
    current_task = state["current_task"]
    return (
        f"**Environment Status:** Active\n\n"
        f"**Current Task:** {state['task_index'] + 1}/{state['total_tasks']}\n"
        f"**Current Score:** {state['total_reward']:.1f}\n\n"
        f"**Current Email:**\n"
        f"**From:** {current_task['sender']}\n"
        f"**Subject:** {current_task['subject']}\n"
        f"**Difficulty:** {current_task['difficulty']}\n\n"
        f"**Task ID:** {current_task['task_id']}"
    )

# Create Gradio interface
with gr.Blocks(title="Email Triage Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📧 Email Triage Environment")
    gr.Markdown("## OpenEnv Round 1 - Smart Email Classification System")
    
    with gr.Tab("🎮 Environment"):
        with gr.Row():
            with gr.Column(scale=2):
                email_display = gr.Markdown("Click 'Reset Environment' to start")
                
            with gr.Column(scale=1):
                reset_btn = gr.Button("🔄 Reset Environment", size="lg", variant="primary")
        
        with gr.Row():
            with gr.Column():
                category_dropdown = gr.Dropdown(
                    choices=["urgent", "normal", "spam"],
                    label="📂 Email Category",
                    info="Classify the email urgency"
                )
                response_dropdown = gr.Dropdown(
                    choices=["reply", "ignore", "escalate"],
                    label="⚡ Response Action",
                    info="Choose the appropriate response"
                )
                submit_btn = gr.Button("📤 Submit Action", size="lg", variant="primary")
        
        with gr.Row():
            with gr.Column():
                result_display = gr.Markdown("Results will appear here")
                reward_display = gr.Number(label="🏆 Current Reward", precision=1)
                done_display = gr.Checkbox(label="✅ Episode Complete", interactive=False)
                status_display = gr.Textbox(label="📊 Status", interactive=False)
    
    with gr.Tab("📊 Environment Info"):
        info_display = gr.Markdown("Environment information will appear here")
        refresh_info_btn = gr.Button("🔄 Refresh Info")
    
    with gr.Tab("📖 Instructions"):
        gr.Markdown("""
        ## How to Use
        
        1. **Reset Environment**: Click the reset button to start with the first email
        2. **Analyze Email**: Read the email content carefully
        3. **Classify**: Choose the appropriate category (urgent/normal/spam)
        4. **Respond**: Select the best response action (reply/ignore/escalate)
        5. **Submit**: Click submit to process your action
        6. **Continue**: Repeat until all 9 emails are processed
        
        ## Scoring System
        
        - **Category Correct**: 0.5 points
        - **Response Correct**: 0.5 points
        - **Total per Email**: 1.0 points maximum
        - **Episode Total**: 9.0 points maximum
        
        ## Task Difficulties
        
        - **Easy** (Tasks 1-3): Simple, clear emails
        - **Medium** (Tasks 4-6): Moderate complexity
        - **Hard** (Tasks 7-9): Complex, urgent scenarios
        
        ## Goal
        
        Achieve the highest possible score by correctly classifying and responding to all emails!
        """)
    
    # Event handlers
    reset_btn.click(
        reset_environment,
        outputs=[email_display, reward_display, done_display, status_display]
    )
    
    submit_btn.click(
        process_action,
        inputs=[category_dropdown, response_dropdown],
        outputs=[email_display, reward_display, done_display, status_display]
    )
    
    refresh_info_btn.click(
        get_environment_info,
        outputs=[info_display]
    )
    
    # Initialize environment info on load
    demo.load(get_environment_info, outputs=[info_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
