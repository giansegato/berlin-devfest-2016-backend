from __future__ import division
import os
from sseclient import SSEClient
import socket
import requests
import json
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# 1st: how can I connect to Firebase: REST APIs
AUTH_TOKEN = "PUWx2kElB4o8rzR74B0mVVscHTshUXG1A1jj3WH8"
FIREBASE_URL = "https://gdg-workshop-9a73b.firebaseio.com/rating_popup"
FIREBASE_ENDING_URL = ".json?auth=" + AUTH_TOKEN

# 4th: data modeling
def processData(data):
    df = pd.DataFrame.transpose(pd.read_json(json.dumps(data)))
    df = df.dropna(subset = [key for key in df.keys() if "x_" in key])
    df = df[pd.notnull(df['y_observed'])]
    
    X = df[[key for key in df.keys() if "x_" in key]].values
    y = df["y_observed"].values

    return X, y

# 5th: initial model
def initialModeling(data):
    X, y = processData(data)

    global n
    n = X.shape[1]

    print "I'm training the model using ", X.shape[0], " samples and ", n, " features.\n"

    global model
    model = SGDClassifier(loss="log", alpha=100, verbose=1)
    model.fit(X, y)

# 6th: update model
def updateModel(path):
    global n
    global model

    parentNode = os.path.split(path)[0]
    j = requests.get(FIREBASE_URL + parentNode + FIREBASE_ENDING_URL).json()
    
    X_sample, y_sample = processData({"sample": j})
    print j

    if (X_sample.shape[1] == n):
        model.partial_fit(X_sample, y_sample)
        print "\n"
    else:
        print "Couldn't update model, wrong dimension: ", X_sample.shape[1], " vs ", n, "\n"

# 7th: predict
def predict(path):
    global n
    global model

    parentNode = os.path.split(path)[0]
    j = requests.get(FIREBASE_URL + parentNode + FIREBASE_ENDING_URL).json()
    sample = np.asarray([j[x] for x in j.keys() if "x_" in x]).reshape(1, -1)

    if sample.shape[1] == n:
        j['y_predicted'] = model.predict_proba(sample)[0][1]
        j['action'] = model.predict(sample)[0]
        r = requests.put(FIREBASE_URL + parentNode + FIREBASE_ENDING_URL, data = json.dumps(j))

        print "Features: ", sample
        print "Prediction: ", model.predict_proba(sample) 
    else:
        print "Couldn't make prediction, wrong dimension: ", sample.shape[1], " vs ", n

# 2nd: SSEClient that prints stuff
if __name__ == '__main__':
    print "Starting SSEClient @ ", FIREBASE_URL + FIREBASE_ENDING_URL, "\n"

    try:
        sse = SSEClient(FIREBASE_URL + FIREBASE_ENDING_URL)
        for msg in sse:
            msg_data = json.loads(msg.data)
            if msg_data is None: # Keep alive
                continue
            elif msg_data['data'] is None: # Deleted node
                continue
            
            #print "\n", msg_data
            
            path = msg_data['path']
            data = msg_data['data']

            # 3rd: handle different cases
            if path == "/":
                print "Initial update: building model\n"
                initialModeling(data)
            elif "x_" in path:
                print "A user updated features: requested prediction\n"
                predict(path)
            elif "y_observed" in path:
                print "A user updated the whole sample: update model\n"
                updateModel(path)
            else:
                print "Unknown action: ", msg_data
            # -- end 3rd --
            
    except socket.error:
        print socket.error
        pass