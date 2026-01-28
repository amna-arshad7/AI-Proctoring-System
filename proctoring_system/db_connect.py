# db_connect.py
from pymongo import MongoClient

def get_db():
    # MongoDB connection URI 
    client = MongoClient("")  
    db = client["exam_proctor"]  
    return db
