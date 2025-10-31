# MLOps Assignment 4: 


## What I Did: I built an automatic code checker using GitHub Actions.

- It uses DVC to get the ML model and data.

- It uses Pytest to run tests and check the model's score.

- It uses CML to post a test report on GitHub.

## Why It's Important: 
- It automatically checks for mistakes, which helps teams build strong software without breaking it.

## What I Learned: 
- I learned how to use GitHub Actions, Pytest, DVC, and CML for automation.

## Challenges and Solutions
- Challenge: The Python installer (pip) kept failing because the virtual computer was missing basic code-building tools.

- Solution: I installed the "build-essential" tools on the computer.

- Challenge: The installer got stuck in loops trying to find the right versions of the libraries.

- Solution: I created a perfect "shopping list" (requirements.txt) and a "recipe" (dvc.yaml) to tell the tools exactly what to do. I also fixed bugs in the Python code so it used the correct column names.

- Challenge: I couldn't log in to Google Drive because the browser was blocked.

- Solution: I created a special robot user (a Service Account) with its own secret key to log in automatically without a browser.
