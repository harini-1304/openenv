import requests
import json

# Test the environment locally
ENV_URL = "http://127.0.0.1:7860"

def test_openenv_spec():
    """Test OpenEnv specification compliance"""
    print("🔍 Testing OpenEnv Specification Compliance")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{ENV_URL}/", timeout=5)
        print(f"✓ Health check: {response.status_code}")
        print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    # Test 2: Reset endpoint
    try:
        response = requests.post(f"{ENV_URL}/reset", timeout=5)
        print(f"✓ Reset endpoint: {response.status_code}")
        reset_data = response.json()
        
        # Check required fields
        required_fields = ["observation", "reward", "done", "info"]
        for field in required_fields:
            if field not in reset_data:
                print(f"✗ Missing field in reset: {field}")
                return False
        
        # Check observation structure
        obs = reset_data["observation"]
        if not isinstance(obs, dict):
            print(f"✗ Observation must be a dict, got {type(obs)}")
            return False
        
        print(f"✓ Reset response structure OK")
        
    except Exception as e:
        print(f"✗ Reset endpoint failed: {e}")
        return False
    
    # Test 3: Step endpoint
    try:
        action_data = {"category": "normal", "response": "reply"}
        response = requests.post(f"{ENV_URL}/step", json=action_data, timeout=5)
        print(f"✓ Step endpoint: {response.status_code}")
        step_data = response.json()
        
        # Check required fields
        required_fields = ["observation", "reward", "done", "info"]
        for field in required_fields:
            if field not in step_data:
                print(f"✗ Missing field in step: {field}")
                return False
        
        print(f"✓ Step response structure OK")
        
    except Exception as e:
        print(f"✗ Step endpoint failed: {e}")
        return False
    
    # Test 4: State endpoint
    try:
        response = requests.get(f"{ENV_URL}/state", timeout=5)
        print(f"✓ State endpoint: {response.status_code}")
        state_data = response.json()
        
        # Check state structure
        if "total_tasks" not in state_data:
            print(f"✗ Missing total_tasks in state")
            return False
        
        print(f"✓ State response structure OK")
        
    except Exception as e:
        print(f"✗ State endpoint failed: {e}")
        return False
    
    print("\n🎉 All OpenEnv specification checks passed!")
    return True

if __name__ == "__main__":
    test_openenv_spec()
