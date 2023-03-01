# spending-balance-example

This repo holds the basic functionality of a machine learning tool I made to automate a part of my families personal finances.

## Problem Statement
My wife and I share many expenses (utilities, mortgage, groceries, pets, etc.), but some expenses we only benefit from as individuals. We have our own credit cards, a joint checking account, and our own checking accounts. Most of our purchases run through our own credit cards. When it is time to pay the credit card bill, we both want to know how much we need to pull from our joint account and our personal accounts to pay in full. This leads to a binary classification problem that can easily be solved with machine learning. 

In this repo, I run the script `run_script.py` to load the trained model, predict which category our various expenditures come from, and total up the amount we need from each account. The transactions are downloaded from CapitalOne's website each month. 

The model has accuracy of about 84%, which is sufficient for us. Occasionally, we will label more data and train a new model to keep up with data drift. (For example, we used to have our own gym memberships, but recently opted for a family membership to save money). 

This saves us each a few minutes each month, and is especially helpful if we forget to transfer money between accounts for a few months. 
