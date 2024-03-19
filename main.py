import tkinter as tk
from tkinter import messagebox
import json
import random

import pandas as pd
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["travel_app"]
collection_user = db["only_users"]
# Query MongoDB and convert result to pandas DataFrame
cursor = collection_user.find()
df = pd.DataFrame(list(cursor))


# Constants
AGE = [18,22,26,29,28,24,21,30,20,33,31,36,35]
HABITS = ["Early-riser"," Grateful"," Hydrated"," Active"," Journaler"," Breather"," Screen-limiter"," Kind"," Break-taker"," Organized"," Hobbyist"," Connector"," Self-caring"," Learner"," Postured"," Nature-lover"," Prompt"," Mindful-eater"," Decaffeinated"," Boundaried"]
ATTITUDE = ["Adventurous"," Analytical"," Assertive"," Authentic"," Charismatic"," Compassionate"," Creative"," Disciplined"," Empathetic"," Enthusiastic"," Flexible"," Focused"," Humble"," Independent"," Innovative"," Intuitive"," Logical"," Optimistic"," Patient"," Perseverant"," Wise"]
BEHAVIOR = ["Assertiveness"," Empathy"," Patience"," Confidence"," Open-mindedness"," Adaptability"," Resilience"," Responsibility"," Communication"," Leadership"," Cooperation"," Self-control"," Integrity"," Accountability"," Flexibility"," Perseverance"," Respect"," Generosity"," Humility"]
SEX = ["Male"," Female"]
ECONOMIC_SITUATION = ["Employed"," Unemployed"," Self-employed"," Retired"," Student"," Underemployed"," Freelancer"," Business owner"," Investor"]

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["travel_app"]
collection = db["only_users"]

# Function to save user to MongoDB
def save_user():
    username = name_var.get()
    age = random.sample(AGE, k=1)
    habits = random.sample(HABITS, k=3)
    attitude =  random.sample(ATTITUDE, k=3)
    behavior =random.sample(BEHAVIOR, k=3)
    econimic_situation = random.sample(ECONOMIC_SITUATION, k=1)
    sex = random.sample(SEX, k=1)
    trip_over = trip_over_var.get()
    self_over = self_over_var.get()




    user_data = {
        "username": username,
        "age": age,
        "habits": habits,
        "attitude": attitude,
        "behavior": behavior,
        "econimic_situation":econimic_situation,
        "sex":  sex,
        "trip_overview": trip_over,
        "self_overview": self_over
    }
    if username == "":
        messagebox.showinfo("Error", "Username can't found")
    else:
        collection.insert_one(user_data)


# Tkinter interface
window = tk.Tk()
window.title("User Data Entry")

tk.Label(window, text="your name:").pack()
name_var = tk.StringVar(window)
name_var.set("")  # Default value
rating_entry = tk.Entry(window, textvariable=name_var)
rating_entry.pack()


tk.Label(window, text="overview about Trips:").pack()
trip_over_var = tk.StringVar(window)
trip_over_var.set("")  # Default value
rating_entry = tk.Entry(window, textvariable=trip_over_var)
rating_entry.pack()

tk.Label(window, text="overview about yourseelf:").pack()
self_over_var = tk.StringVar(window)
self_over_var.set("")  # Default value
rating_entry = tk.Entry(window, textvariable=self_over_var)
rating_entry.pack()




# Save button
save_button = tk.Button(window, text="Save", command=save_user)
save_button.pack()

window.mainloop()
