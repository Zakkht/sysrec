import tkinter as tk
from tkinter import messagebox
import json
import random

import pandas as pd
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["travel_app"]
collection = db["only_users"]
# Query MongoDB and convert result to pandas DataFrame
cursor = collection.find()
df = pd.DataFrame(list(cursor))
users = df['username'].unique()

# Constants



rating = [1,2,3,4,5]



# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["travel_app"]
collection = db["user rating"]

# Function to save user to MongoDB
def save_user():
    username = user_var.get()
    username2 = user_var2.get()
    rating = rating_var.get()




    user_data = {
        "username": username,
        "username2": username2,
        "rating": rating
    }
    if username == "":
        messagebox.showinfo("Error", "Username can't found")
    else:
        collection.insert_one(user_data)


# Tkinter interface
window = tk.Tk()
window.title("User Data Entry")

# Username
tk.Label(window, text="username:").pack()
user_var = tk.StringVar(window)
user_var.set(users[0])  # Default value
user_menu = tk.OptionMenu(window, user_var, *users)
user_menu.pack()


# Username2
tk.Label(window, text="username2:").pack()
user_var2 = tk.StringVar(window)
user_var2.set(users[0])  # Default value
user_menu = tk.OptionMenu(window, user_var2, *users)
user_menu.pack()


# Rating
tk.Label(window, text="Rating (1-5):").pack()
rating_var = tk.IntVar(window)
rating_var.set(rating[0])  # Default value
rating_entry = tk.OptionMenu(window, rating_var,*rating)
rating_entry.pack()

# Save button
save_button = tk.Button(window, text="Save", command=save_user)
save_button.pack()

window.mainloop()
