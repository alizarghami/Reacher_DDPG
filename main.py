# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:07:16 2020

@author: Ali
"""

from reacher import Reacher


# please adjust the parameters below
env_path = "Reacher_Windows_x86_64/Reacher.exe"

# Create a navigation instance
r = Reacher(env_path, criteria=30, seed=0)

# Load the pre-trained model
try:
#    r.load_model()
    print("Model loaded successfully")
except:
    print("No model to load...")

# Train the model
outcome = r.train()

# Save the trained model if the criteria is reached
if outcome:
    r.save_model()

if outcome:
    print("Criteria reached...")
    # Evaluate the model
    r.evaluate()

# Close the unity environment
r.close_env()