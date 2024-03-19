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



# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["travel_app"]
collection = db["trip_infos"]

# Function to save user to MongoDB
def save_user():
    place = name_var.get()
    overview = overview_var.get()




    user_data = {
        "place": place,
        "overview": overview,

    }
    if place == "":
        messagebox.showinfo("Error", "Username can't found")
    else:
        collection.insert_one(user_data)


# Tkinter interface
window = tk.Tk()
window.title("User Data Entry")

tk.Label(window, text="place:").pack()
name_var = tk.StringVar(window)
name_var.set("")  # Default value
rating_entry = tk.Entry(window, textvariable=name_var)
rating_entry.pack()

tk.Label(window, text="your overview about user:").pack()
overview_var = tk.StringVar(window)
overview_var.set("")  # Default value
rating_entry = tk.Entry(window, textvariable=overview_var)
rating_entry.pack()




# Save button
save_button = tk.Button(window, text="Save", command=save_user)
save_button.pack()

window.mainloop()
