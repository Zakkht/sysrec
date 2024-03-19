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
users = df['username'].unique()

# Constants
PLACES = ["Paris", "Tokyo", "New York City", "Sydney", "Rio de Janeiro", "Cape Town", "Dubai", "Barcelona", "Bali", "Vancouver"]
SEASONS = ["winter", "fall", "spring", "summer"]
TRANSPORTS = ["car", "bike", "bus", "train", "airplane"]
rating = [1,2,3,4,5]
ACTIVITIES = {
    "Paris": ["Cafe", "Museums", "Parks", "Markets", "Theatre"],
    "Tokyo": ["Temples", "Gardens", "Anime", "Sumo", "Shopping"],
    "New York City": ["Theater", "Music", "Art", "Parks", "Food"],
    "Sydney": ["Beaches", "Hiking", "Wildlife", "Surfing", "Sailing"],
    "Rio de Janeiro": ["Samba", "Beaches", "Hiking", "Markets", "Streetfood"],
    "Cape Town": ["Hiking", "Wildlife", "Beaches", "Wine", "Markets"],
    "Dubai": ["Shopping", "Desert", "Beaches", "Skyscrapers", "Cuisine"],
    "Barcelona": ["Architecture", "Beaches", "Tapas", "Markets", "Parks"],
    "Bali": ["Yoga", "Surfing", "Temples", "Markets", "Ricefields"],
    "Vancouver": ["Mountains", "Parks", "Biking", "Beaches", "Skiing"]
}
GEOGRAPHIES = {
    "Paris": ["Riverine", "Flat", "Gardens"],
    "Tokyo": ["Mountainous", "Coastal", "Urban"],
    "New York City": ["Coastal", "Islands", "Urban"],
    "Sydney": ["Harbor", "Coastal", "National Parks"],
    "Rio de Janeiro": ["Coastal", "Mountainous", "Rainforest"],
    "Cape Town": ["Mountainous", "Coastal", "Winelands"],
    "Dubai": ["Desert", "Coastal", "Skyscrapers"],
    "Barcelona": ["Coastal", "Urban", "Mountains"],
    "Bali": ["Volcanic", "Beaches", "Rice Terraces"],
    "Vancouver": ["Coastal", "Mountainous", "Rainforest"]
}

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["travel_app"]
collection = db["users"]

# Function to save user to MongoDB
def save_user():
    username = user_var.get()
    place = place_var.get()
    season = season_var.get()
    transport = transport_var.get()
    activities = random.sample(ACTIVITIES[place], k=3)
    geography = GEOGRAPHIES[place]
    rating = rating_var.get()




    user_data = {
        "username": username,
        "place": place,
        "season": season,
        "transport": transport,
        "activities": activities,
        "geography": geography,
        "rating": rating
    }
    if username == "":
        messagebox.showinfo("Error", "Username can't found")
    else:
        collection.insert_one(user_data)
        messagebox.showinfo("Success", "User data saved to MongoDB")

# Tkinter interface
window = tk.Tk()
window.title("User Data Entry")

# Username
tk.Label(window, text="username:").pack()
user_var = tk.StringVar(window)
user_var.set(users[0])  # Default value
user_menu = tk.OptionMenu(window, user_var, *users)
user_menu.pack()


# Place
tk.Label(window, text="Place:").pack()
place_var = tk.StringVar(window)
place_var.set(PLACES[0])  # Default value
place_menu = tk.OptionMenu(window, place_var, *PLACES)
place_menu.pack()

# Season
tk.Label(window, text="Season:").pack()
season_var = tk.StringVar(window)
season_var.set(SEASONS[0])  # Default value
season_menu = tk.OptionMenu(window, season_var, *SEASONS)
season_menu.pack()

# Transport
tk.Label(window, text="Transport:").pack()
transport_var = tk.StringVar(window)
transport_var.set(TRANSPORTS[0])  # Default value
transport_menu = tk.OptionMenu(window, transport_var, *TRANSPORTS)
transport_menu.pack()

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
