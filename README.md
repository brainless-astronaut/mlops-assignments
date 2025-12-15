# Week 11: LLM Governance and Guardrails

## Student Details
**ID:** 21F3002750
**Name:** S Sahana

## Project Overview
This project evaluates the security vulnerabilities of a simulated Fine-Tuned LLM pipeline and implements a Guardrail layer to mitigate risks.

## Files Included
1. **llm_pipeline.py**: The main inference pipeline using `distilgpt2`. It mimics a finetuned model and integrates the governance layer.
2. **guardrails.py**: A custom Python class that implements Input/Output filtering. It detects:
   - Prompt Injection (e.g., "Ignore previous instructions")
   - PII/Secret Leakage (e.g., "API Key")
   - Harmful Content (e.g., "Dangerous weapon")
3. **evaluate_governance.py**: An evaluation script that runs a battery of adversarial prompts against the model—first without guardrails, then with guardrails enabled—and saves the results to CSV.
4. **unsafe_results.csv**: Evidence of the model's vulnerability before governance.
5. **safe_results.csv**: Evidence of the guardrails successfully blocking attacks.

## How to Run
1. Install dependencies: `pip install transformers torch pandas`
2. Run the evaluation: `python evaluate_governance.py`
3. Check the generated CSV files for the audit report.
