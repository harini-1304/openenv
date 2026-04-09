import os
import requests
import json
import time
from typing import Dict, Any, Optional

print("=== INFERENCE.PY STARTED ===")
print("=== THIS IS THE ULTRA DEBUG VERSION ===")
print("=== IF YOU SEE THIS, INFERENCE.PY IS RUNNING ===")
print(f"Python version: {os.sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Constants
ENVIRONMENT_URL = "https://harini-1304-email-triage-env-final.hf.space"
TASK_NAME = "email_triage"
BENCHMARK = "openenv_round1"
MAX_STEPS = 9
TEMPERATURE = 0.1
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1

def load_environment_variables():
    """Load and validate environment variables"""
    print("[ULTRA DEBUG] All available environment variables:")
    for key, value in sorted(os.environ.items()):
        if any(keyword in key.upper() for keyword in ['API', 'KEY', 'TOKEN', 'URL', 'MODEL', 'OPENENV']):
            if 'KEY' in key.upper() or 'TOKEN' in key.upper():
                print(f"  {key}: *** ({len(value)} chars)")
            else:
                print(f"  {key}: {value}")

    print("[ULTRA DEBUG] Attempting to load required variables...")

    # Use OpenEnv's actual variable names
    try:
        API_BASE_URL = os.environ.get("OPENAI_API_BASE_URL") or os.environ.get("HF_ENDPOINT", "https://api.openai.com/v1")
        print(f"[ULTRA DEBUG] API_BASE_URL loaded: {API_BASE_URL}")
    except Exception as e:
        print(f"[ULTRA DEBUG] API_BASE_URL ERROR: {e}")
        print("[ULTRA DEBUG] Available URL-like variables:")
        for key, value in os.environ.items():
            if 'URL' in key.upper():
                print(f"  {key}: {value}")
        # Continue with default if missing
        API_BASE_URL = API_BASE_URL or "https://api.openai.com/v1"

    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    print(f"[ULTRA DEBUG] MODEL_NAME: {MODEL_NAME}")

    try:
        API_KEY = os.environ["OPENAI_API_KEY"]  # Use OpenEnv's variable name
        print(f"[ULTRA DEBUG] API_KEY loaded: {len(API_KEY)} chars, starts with: {API_KEY[:10]}...")
    except KeyError as e:
        print(f"[ULTRA DEBUG] API_KEY MISSING: {e}")
        print("[ULTRA DEBUG] Available KEY-like variables:")
        for key, value in os.environ.items():
            if 'KEY' in key.upper() or 'TOKEN' in key.upper():
                print(f"  {key}: *** ({len(value)} chars)")
        # Fallback to HF_TOKEN if OPENAI_API_KEY missing
        API_KEY = API_KEY or os.environ.get("HF_TOKEN", "")
        if not API_KEY:
            raise ValueError("No API key found in OPENAI_API_KEY or HF_TOKEN")

    print(f"[ULTRA DEBUG] All environment variables loaded successfully")
    print(f"[ULTRA DEBUG] Final API_BASE_URL: {API_BASE_URL}")
    print(f"[ULTRA DEBUG] Final API_KEY: {len(API_KEY)} chars")
    
    return API_BASE_URL, MODEL_NAME, API_KEY

class EmailTriageAgent:
    def __init__(self, api_base_url, model_name, api_key):
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.api_key = api_key
        self.total_reward = 0.0
        self.episodes_completed = 0
        self.step_count = 0
        self.rewards = []
        
    def classify_email_llm(self, email: str) -> Dict[str, str]:
        """Use LLM to classify email"""
        try:
            import openai
            print(f"[ULTRA DEBUG] About to import openai")
            # Use exact OpenEnv format
            print(f"[ULTRA DEBUG] Initializing OpenAI client")
            print(f"[ULTRA DEBUG] Base URL: {self.api_base_url}")
            print(f"[ULTRA DEBUG] API Key: {len(self.api_key)} chars")
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url
            )
            print(f"[ULTRA DEBUG] OpenAI client created successfully")
            
            print(f"[ULTRA DEBUG] About to make API call")
            print(f"[ULTRA DEBUG] Model: {self.model_name}")
            print(f"[ULTRA DEBUG] Email length: {len(email)} chars")
            print(f"[DEBUG] Model: {self.model_name}")
            print(f"[DEBUG] Temperature: {TEMPERATURE}")
            print(f"[DEBUG] Max tokens: {MAX_TOKENS}")
            prompt = f"""
You MUST respond ONLY in valid JSON.

Format:
{{"category":"urgent|normal|spam","response":"reply|ignore|escalate"}}

Email:
{email}
"""
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an email triage expert. Classify emails and determine appropriate responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            print(f"[ULTRA DEBUG] API call completed successfully!")
            print(f"[ULTRA DEBUG] Response type: {type(response)}")
            print(f"[ULTRA DEBUG] Response choices count: {len(response.choices)}")
            print(f"[ULTRA DEBUG] First choice: {response.choices[0]}")
            print(f"[ULTRA DEBUG] Message content: {response.choices[0].message.content}")
            print("[LLM RAW OUTPUT]:", response.choices[0].message.content)
            
            # Extract JSON safely
            content = response.choices[0].message.content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            
            if start != -1 and end != -1:
                json_str = content[start:end]
                result = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
            
            print(f"[DEBUG] Parsed LLM result: {result}")
            return result
            
        except Exception as e:
            print(f"[ERROR] LLM FAILED:", e)
            import traceback
            traceback.print_exc()
            # CRITICAL: Do not return fallback - force the error to be visible
            raise Exception(f"LLM API call failed: {e}")
    
    def classify_email_rules(self, email: str) -> Dict[str, str]:
        """Rule-based email classification"""
        email_lower = email.lower()
        
        # Urgent keywords
        urgent_keywords = ["urgent", "emergency", "critical", "immediate", "asap", "down", "failure", "corruption"]
        
        # Spam keywords
        spam_keywords = ["free", "prize", "winner", "claim", "discount", "offer", "special", "limited time"]
        
        # Check for urgency
        if any(keyword in email_lower for keyword in urgent_keywords):
            if any(keyword in email_lower for keyword in ["server", "database", "system", "infrastructure"]):
                return {"category": "urgent", "response": "escalate", "reasoning": "Technical emergency detected"}
            else:
                return {"category": "urgent", "response": "reply", "reasoning": "Urgent business matter"}
        
        # Check for spam
        elif any(keyword in email_lower for keyword in spam_keywords):
            return {"category": "spam", "response": "ignore", "reasoning": "Spam indicators detected"}
        
        # Default to normal
        else:
            return {"category": "normal", "response": "reply", "reasoning": "Standard business email"}
    
    def reset_environment(self) -> Dict[str, Any]:
        """Reset the environment"""
        try:
            response = requests.post(f"{ENVIRONMENT_URL}/reset", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[ERROR] Reset failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"[ERROR] Reset request failed: {e}")
            return None
    
    def submit_action(self, category: str, response: str) -> Dict[str, Any]:
        """Submit action to environment"""
        try:
            action_data = {"category": category, "response": response}
            response = requests.post(f"{ENVIRONMENT_URL}/step", json=action_data, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[ERROR] Action failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"[ERROR] Action request failed: {e}")
            return None
    
    def log_start(self, task: str, env: str, model: str) -> None:
        """Log episode start"""
        print(f"[START] task={task} env={env} model={model}", flush=True)
    
    def log_step(self, step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
        """Log step with required format"""
        error_val = error if error else "null"
        done_val = str(done).lower()
        print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)
    
    def log_end(self, success: bool, steps: int, score: float, rewards: list) -> None:
        """Log episode end with required format"""
        success_val = str(success).lower()
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)
    
    def run_episode(self) -> float:
        """Run one complete episode with proper logging"""
        self.log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
        
        # Reset environment
        reset_result = self.reset_environment()
        if not reset_result:
            print("[ERROR] Failed to reset environment")
            self.log_end(False, 0, 0.0, [])
            return 0.0
        
        episode_reward = 0.0
        self.step_count = 0
        self.rewards = []
        episode_success = True
        
        while not reset_result.get("done", False) and self.step_count < MAX_STEPS:
            self.step_count += 1
            
            # Get current email
            observation = reset_result.get("observation", {})
            email = observation.get("email", "")
            task_id = observation.get("task_id", 0)
            
            # Classify email
            classification = self.classify_email_llm(email)
            category = classification.get("category", "normal")
            response = classification.get("response", "reply")
            
            # Format action string
            action_str = f"category={category},response={response}"
            
            # Submit action
            step_result = self.submit_action(category, response)
            if not step_result:
                print(f"[ERROR] Failed to submit action for task {task_id}")
                self.log_step(self.step_count, action_str, 0.0, False, "Failed to submit action")
                episode_success = False
                break
            
            # Get reward
            reward = step_result.get("reward", 0.0)
            episode_reward += reward
            self.rewards.append(reward)
            
            # Check if episode is done
            done = step_result.get("done", False)
            
            # Log step
            self.log_step(self.step_count, action_str, reward, done, None)
            
            # Update for next iteration
            reset_result = step_result
        
        # Calculate final score (normalized to [0, 1])
        final_score = min(episode_reward / 9.0, 1.0)  # Normalize to [0, 1]
        
        # Log episode end
        self.log_end(episode_success, self.step_count, final_score, self.rewards)
        
        self.total_reward += episode_reward
        self.episodes_completed += 1
        
        return final_score

def main():
    """Main inference function with proper logging"""
    print("=== Email Triage Environment Inference ===")
    
    # Load environment variables safely
    try:
        API_BASE_URL, MODEL_NAME, API_KEY = load_environment_variables()
        print(f"API Base URL: {API_BASE_URL}")
        print(f"Model: {MODEL_NAME}")
        print(f"Environment URL: {ENVIRONMENT_URL}")
        print()
        
        agent = EmailTriageAgent(API_BASE_URL, MODEL_NAME, API_KEY)
    except Exception as e:
        print(f"[ERROR] Failed to load environment variables: {e}")
        print("[END] success=false steps=0 score=0.00 rewards=")
        return
    
    try:
        # Test environment connectivity first
        test_response = requests.get(f"{ENVIRONMENT_URL}/", timeout=5)
        if test_response.status_code != 200:
            print(f"[ERROR] Environment not reachable: {test_response.status_code}")
            return
        
        # Run multiple episodes
        num_episodes = 3
        total_score = 0.0
        
        for episode in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"EPISODE {episode + 1}/{num_episodes}")
            print(f"{'='*50}")
            
            episode_score = agent.run_episode()
            total_score += episode_score
            
            print(f"Episode {episode + 1} score: {episode_score:.2f}")
            
            # Small delay between episodes
            if episode < num_episodes - 1:
                print("Waiting 3 seconds before next episode...")
                time.sleep(3)
        
        # Final results
        average_score = total_score / num_episodes
        print(f"\n{'='*50}")
        print("FINAL RESULTS")
        print(f"{'='*50}")
        print(f"Total episodes: {num_episodes}")
        print(f"Average score: {average_score:.2f}/1.00")
        print(f"Average accuracy: {average_score*100:.1f}%")
        print(f"Total reward accumulated: {agent.total_reward:.1f}")
        
        # Performance rating
        if average_score >= 0.90:
            rating = "EXCEPTIONAL"
        elif average_score >= 0.75:
            rating = "EXCELLENT"
        elif average_score >= 0.60:
            rating = "GOOD"
        else:
            rating = "NEEDS IMPROVEMENT"
        
        print(f"Performance Rating: {rating}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Inference interrupted by user")
        print("[END] success=false steps=0 score=0.00 rewards=")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network request failed: {e}")
        print("[END] success=false steps=0 score=0.00 rewards=")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        print("[END] success=false steps=0 score=0.00 rewards=")

if __name__ == "__main__":
    print("=== ABOUT TO CALL MAIN() ===")
    main()
    print("=== MAIN() COMPLETED ===")
