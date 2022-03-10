from os import remove
import pyrebase
import time
import json
from datetime import datetime
  

firebaseConfig = {"apiKey": "AIzaSyBLYegf7GSv4TFRv0d2q2AY0NUhMJH9YJ8",
  "authDomain": "doorlock-bb886.firebaseapp.com",
  "databaseURL": "https://doorlock-bb886-default-rtdb.asia-southeast1.firebasedatabase.app/",
  "projectId": "doorlock-bb886",
  "storageBucket": "doorlock-bb886.appspot.com",
  "messagingSenderId": "962039315770",
  "appId": "1:962039315770:web:573e203a01c9dfe0fce403",
  "measurementId": "G-5ZX3C4S204"}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

# for i in range(10):
#     data = {"name": i, "age:": "20"}
#     db.child("test").push(data)

# items = db.child("test").order_by_child("name").limit_to_last(3).get()
# for item in items.each():
#     print(item.val())

def getFBData(number):
    items = db.child("history").order_by_child("timestamp").limit_to_last(number).get()
    datas = getStrData(items)
    datas.sort(reverse=True)
    return datas
def putData(name, opened, path):
    timestamp = round(time.time() * 1000)
    link = putImage(path)
    data = {
        "timestamp": timestamp,
        "name": name,
        "opened": opened,
        "link": link
    }
    db.child("history").push(data)

storage = firebase.storage()

def putImage(path):
    timestamp = round(time.time() * 1000)
    storage.child(str(timestamp) + ".jpg").put(path)
    # remove(path)
    return storage.child(str(timestamp) + ".jpg").get_url(None)

def getStrData(items):
    a = []
    for item in items.each():
        y = item.val()
        x = [getTime(y["timestamp"]), y["name"], y["opened"], y["link"]]
        a.append(x)
    return a

def getTime(timestamp):
    dt_obj = datetime.fromtimestamp(timestamp//1000)
    return dt_obj

# if __name__ == '__main__':
#     items = getData(10)
#     for item in items.each():
#         y = item.val()
#         print(y["link"])
