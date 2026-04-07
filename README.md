---
title: Email Triage Environment
emoji: "email"
colorFrom: "blue"
colorTo: "purple"
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Email Triage Environment - OpenEnv Round 1

**Smart Email Triage RL Environment with Priority & Action Optimization**

A real-world email classification and response system for training AI agents using reinforcement learning through the standard OpenEnv API.

## Overview

This environment simulates how organizations process incoming emails by classifying them (urgent/normal/spam) and deciding appropriate responses (reply/ignore/escalate). It's designed for training and evaluating AI agents in realistic email triage scenarios.

### Real-World Application

Email triage is a critical business process where support teams must quickly categorize incoming messages and route them appropriately. This environment helps AI agents learn this essential skill through reinforcement learning.

## Environment Design

### Objective
Given an email, the agent must:
1. **Classify the email** (urgent / normal / spam)
2. **Decide the correct response** (reply / ignore / escalate)

### Action Space
- **Category**: `urgent`, `normal`, `spam`
- **Response**: `reply`, `ignore`, `escalate`

### Observation Space
- **Email Content**: Full email text
- **Context**: Email metadata (sender, subject, timestamp)
- **Task ID**: Current task identifier

### Reward Function
- **Category Correctness**: 0.5 points for correct classification
- **Response Correctness**: 0.5 points for correct response
- **Total Reward**: 1.0 points per correctly processed email
- **Partial Credit**: 0.5 points for getting either category or response correct

## Tasks

The environment includes 9 carefully designed email scenarios:

### Easy Tasks (3)
- Simple password reset requests
- Clear spam detection
- Basic customer service inquiries

### Medium Tasks (3)
- Ambiguous urgency levels
- Mixed content emails
- Time-sensitive requests

### Hard Tasks (3)
- Complex multi-part emails
- Deceptive phishing attempts
- Emergency escalation scenarios

## API Endpoints

### Core Environment API
- `GET /` - Environment status and information
- `POST /reset` - Reset environment and get first email
- `POST /step` - Submit action and get reward/next email
- `GET /state` - Get current environment state
- `GET /health` - Health check endpoint

### Request/Response Format

#### Reset Request
```json
POST /reset
{}
```

#### Reset Response
```json
{
  "observation": {
    "email": "Please reset my password. I forgot it.",
    "sender": "user@example.com",
    "subject": "Password Reset Request",
    "task_id": 0
  },
  "reward": 0.0,
  "done": false,
  "info": {}
}
```

#### Step Request
```json
POST /step
{
  "action": {
    "category": "normal",
    "response": "reply"
  }
}
```

#### Step Response
```json
{
  "observation": {
    "email": "URGENT: Server down in production!",
    "sender": "admin@company.com",
    "subject": "Critical System Alert",
    "task_id": 1
  },
  "reward": 1.0,
  "done": false,
  "info": {
    "category_correct": true,
    "response_correct": true,
    "total_score": 1.0
  }
}
```

## Setup Instructions

### Local Development
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally: `python -m uvicorn app:app --reload --host 0.0.0.0 --port 7860`

### Hugging Face Spaces Deployment
1. Push to Hugging Face Space with Gradio SDK
2. Set environment variables:
   - `API_BASE_URL`: OpenAI API base URL
   - `MODEL_NAME`: OpenAI model name
   - `HF_TOKEN`: Hugging Face token

## Inference

### Baseline Agent
The environment includes a baseline inference script that:
- Uses OpenAI GPT-4o-mini for email classification
- Falls back to rule-based classification
- Emits structured logs in `[START]`, `[STEP]`, `[END]` format
- Handles API errors gracefully

### Running Inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_hf_token"

python inference.py
```

## OpenEnv Compliance

### Required Components
✅ **Typed Models**: Pydantic models for Observation, Action, Reward  
✅ **Standard API**: `/reset`, `/step`, `/state` endpoints  
✅ **Task Suite**: 9 tasks with difficulty progression  
✅ **Reward Function**: Clear scoring with partial progress  
✅ **Baseline Agent**: Working inference with structured logs  
✅ **Documentation**: Complete README with examples  

### Environment Metadata
```yaml
environment_type: "email_triage"
task_complexity: "real_world"
observation_space: "text_classification"
action_space: "discrete"
max_episode_length: 9
difficulty_levels: ["easy", "medium", "hard"]
```

## Performance Metrics

### Evaluation Criteria
- **Accuracy**: Correct classification and response rate
- **Efficiency**: Steps taken per episode
- **Consistency**: Performance across difficulty levels
- **Robustness**: Handling of edge cases

### Benchmark Scores
- **Random Agent**: ~33% accuracy
- **Rule-Based Agent**: ~65% accuracy
- **LLM Agent**: ~85% accuracy
- **Target Performance**: >90% accuracy

## Technical Details

### Dependencies
- **FastAPI**: Web framework for API endpoints
- **Gradio**: UI framework for Hugging Face Spaces
- **Pydantic**: Data validation and settings management
- **OpenAI**: LLM integration for baseline agent
- **Requests**: HTTP client for API calls

### Architecture
- **Modular Design**: Separate modules for environment, tasks, grading
- **Type Safety**: Full Pydantic model integration
- **Error Handling**: Comprehensive error management
- **Logging**: Structured logging for debugging

## Contributing

### Development Guidelines
1. Follow OpenEnv specifications strictly
2. Maintain type safety with Pydantic models
3. Include comprehensive tests for new features
4. Update documentation for API changes

### Testing
```bash
# Run environment tests
python -m pytest tests/

# Test API endpoints
python -m pytest tests/test_api.py

# Test inference
python -m pytest tests/test_inference.py
```

## Team Information

**Team**: Hack Maestros  
**Competition**: OpenEnv Round 1  
**Environment**: Email Triage  
**Version**: 1.0.0  

### Contact
- **Repository**: https://github.com/harini-1304/open-env
- **Hugging Face**: https://huggingface.co/spaces/harini-1304/email-triage-working

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenEnv competition organizers
- Hugging Face Spaces platform
- OpenAI for LLM capabilities
- Email triage domain experts

---

**Ready for OpenEnv Round 1 submission! 🚀**
