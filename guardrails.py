import re

class SafetyGuardrail:
    def __init__(self):
        self.injection_keywords = ["ignore previous instructions", "system prompt", "delete all"]
        self.leakage_keywords = ["api key", "password", "secret"]
        self.forbidden_topics = ["weapon", "dangerous", "curse"]

    def check_input(self, prompt):
        prompt_lower = prompt.lower()
        
        # Check for Injection
        for keyword in self.injection_keywords:
            if keyword in prompt_lower:
                return False, "Guardrail Alert: Prompt Injection detected."
        
        # Check for Leakage attempts
        for keyword in self.leakage_keywords:
            if keyword in prompt_lower:
                return False, "Guardrail Alert: Potential Information Leakage detected."
                
        # Check for Harmful topics
        for keyword in self.forbidden_topics:
            if keyword in prompt_lower:
                return False, "Guardrail Alert: Harmful content detected."
                
        return True, "Safe"

    def check_output(self, response):
        # Basic PII scrubbing or output filtering
        if "password" in response.lower() or "key" in response.lower():
            return False, "Guardrail Alert: Output contained sensitive info."
        return True, "Safe"
