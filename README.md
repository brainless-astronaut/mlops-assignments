# MLOps Graded Assignment - Week 4 Resources 

Please find the Git command reference from lectures

# Part A: Convert week 2’s code base into a GitHub repo
## Check local status
git status
## Ensure commits are latest
git log
## check user who is logged in locally
git config user.name
git config user.email
## create a new repo in your public GitHub account
## Copy the url this gives you
https://github.com/\<user_name\>/\<repo_name\>.git

## point local to your repo
git remote remove origin

git remote add origin https://github.com/\<user_name\>/\<repo_name\>.git

## configure tokens
Go to github.com/settings/tokens
Create a classic token. Use that instead of a password.

## push code to repo
git push -u origin master

# Part B: Write a simple Test that can be invoked automatically for sanity.

For sanity testing using predictions on samples, the trained
model itself needed to be saved.

Write a Python unit test file. 
Execute to verify - If when the code is attempted to be committed and you see
that HEAD is disconnected, then do the following.

## Then, check out a new branch “unit-tested-branch”
git checkout -b unit-tested-branch

## Confirm that remote is configured correctly
git remote -v

## push branch to remote
git push -u origin unit-tested-branch

## merge branch to master using
git checkout master

git pull origin master

git merge unit-tested-branch

git push origin master
## and then delete the branch both locally and remotely
git branch -d unit-tested-branch

git push origin --delete unit-tested-branch

# Part C: Configure GitHub action to run this test every time code changes.

Follow the yaml.

Once fully done, execute the workflow from Github UI. 

Test out the successful execution.
A full run includes giving permissions to write too. 

Once completed, the job itself shows description of results as a comment under that commit.