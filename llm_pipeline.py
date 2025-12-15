from transformers import pipeline
from guardrails import SafetyGuardrail

class LLMPipeline:
    def __init__(self):
        print("Loading model... (Simulating Finetuned LLM)")
        # Using distilgpt2 because it is smaller/faster than gpt2
        self.generator = pipeline('text-generation', model='distilgpt2')
        self.guard = SafetyGuardrail()

    def generate(self, prompt):
        # Ungoverned generation
        try:
            response = self.generator(prompt, max_length=50, num_return_sequences=1)
            return response[0]['generated_text']
        except Exception as e:
            return str(e)

    def generate_guarded(self, prompt):
        # 1. Input Guardrail
        is_safe, message = self.guard.check_input(prompt)
        if not is_safe:
            return f"[BLOCKED] {message}"

        # 2. Generate
        try:
            response = self.generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        except Exception as e:
            return str(e)

        # 3. Output Guardrail
        is_safe_out, message_out = self.guard.check_output(response)
        if not is_safe_out:
            return f"[BLOCKED] {message_out}"

        return response
