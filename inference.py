import os
import requests
import json
import time
from typing import Dict, Any

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENVIRONMENT_URL = "https://harini-1304-email-triage-env-final.hf.space"

class EmailTriageAgent:
    def __init__(self):
        self.total_reward = 0.0
        self.episodes_completed = 0
        
    def classify_email_llm(self, email: str) -> Dict[str, str]:
        """Use LLM to classify email"""
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""), base_url=API_BASE_URL)
            
            prompt = f"""
Classify this email for triage:

Email: {email}

Respond with JSON format:
{{
    "category": "urgent" or "normal" or "spam",
    "response": "reply" or "ignore" or "escalate",
    "reasoning": "brief explanation"
}}
"""
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an email triage expert. Classify emails and determine appropriate responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"[ERROR] LLM classification failed: {e}")
            return self.classify_email_rules(email)
    
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
    
    def run_episode(self) -> float:
        """Run one complete episode"""
        print("[START] Starting new episode")
        
        # Reset environment
        reset_result = self.reset_environment()
        if not reset_result:
            print("[ERROR] Failed to reset environment")
            return 0.0
        
        episode_reward = 0.0
        step_count = 0
        
        while not reset_result.get("done", False):
            step_count += 1
            
            # Get current email
            observation = reset_result.get("observation", {})
            email = observation.get("email", "")
            task_id = observation.get("task_id", 0)
            
            print(f"[STEP] Processing task {task_id}")
            print(f"[STEP] Email: {email[:100]}...")
            
            # Classify email
            classification = self.classify_email_llm(email)
            category = classification.get("category", "normal")
            response = classification.get("response", "reply")
            reasoning = classification.get("reasoning", "")
            
            print(f"[STEP] Classification: {category}/{response}")
            print(f"[STEP] Reasoning: {reasoning}")
            
            # Submit action
            step_result = self.submit_action(category, response)
            if not step_result:
                print("[ERROR] Failed to submit action")
                break
            
            # Get reward
            reward = step_result.get("reward", 0.0)
            episode_reward += reward
            
            info = step_result.get("info", {})
            category_correct = info.get("category_correct", False)
            response_correct = info.get("response_correct", False)
            
            print(f"[STEP] Step reward: {reward:.1f}")
            print(f"[STEP] Category correct: {category_correct}")
            print(f"[STEP] Response correct: {response_correct}")
            
            # Check if episode is done
            if step_result.get("done", False):
                break
            
            # Update for next iteration
            reset_result = step_result
        
        self.total_reward += episode_reward
        self.episodes_completed += 1
        
        print(f"[END] Episode completed")
        print(f"[END] Episode reward: {episode_reward:.1f}")
        print(f"[END] Total reward: {self.total_reward:.1f}")
        print(f"[END] Episodes completed: {self.episodes_completed}")
        
        return episode_reward

def main():
    """Main inference function"""
    print("=== Email Triage Environment Inference ===")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment URL: {ENVIRONMENT_URL}")
    print()
    
    agent = EmailTriageAgent()
    
    try:
        # Run multiple episodes
        num_episodes = 3
        total_score = 0.0
        
        for episode in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"EPISODE {episode + 1}/{num_episodes}")
            print(f"{'='*50}")
            
            episode_score = agent.run_episode()
            total_score += episode_score
            
            print(f"Episode {episode + 1} score: {episode_score:.1f}/9.0")
            
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
        print(f"Average score: {average_score:.1f}/9.0")
        print(f"Average accuracy: {(average_score/9.0)*100:.1f}%")
        print(f"Total reward accumulated: {agent.total_reward:.1f}")
        
        # Performance rating
        accuracy = (average_score / 9.0) * 100
        if accuracy >= 90:
            rating = "EXCEPTIONAL"
        elif accuracy >= 75:
            rating = "EXCELLENT"
        elif accuracy >= 60:
            rating = "GOOD"
        else:
            rating = "NEEDS IMPROVEMENT"
        
        print(f"Performance Rating: {rating}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Inference interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")

if __name__ == "__main__":
    main()
