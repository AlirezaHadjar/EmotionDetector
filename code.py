from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logger
import numpy as np
import sys
import json
import time
import multiprocessing
import pymongo

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-digikala")
model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-digikala")

def softmax(inputs, all):
    all_ = [float(all[0]), float(all[1]), float(all[2])]
    return np.exp(float(inputs)) / np.sum(np.exp(all_))

def process_text(text):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        logits = (model(**inputs).logits)[0]

    result = {
        "text": text,
        "emotions": {
            "نامشخص": softmax(logits[0], logits),
            "منفی": softmax(logits[1], logits),
            "مثبت":softmax(logits[2], logits)
        }
    }

    return result

def emotion_recognition(docs):
    before = time.time() * 1000

    results = []
    for i, doc in enumerate(docs):
        text = doc['text']
        result = process_text(text)
        results.append(result)
        print("Processed", i + 1, "items", end="\r")

    after = time.time() * 1000
    print("Took", int((after - before) / 1000), "seconds to process", "many", "items")

    return results

if __name__ == '__main__':
    # multiprocessing.freeze_support()
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["tweets"]
    collection = db["mycollection"]
    while True:
        docs = collection.find({
            'processed': False,
            'error': False,
            "$expr": { '$lt': [{ '$strLenCP': "$text" }, 500] }}).limit(2000)
        # cut the docs in half and send them to two different processes
        docs = list(docs)

        if len(docs) == 0:
            break

        half = int(len(docs) / 2)
        docs1 = docs[:half]
        docs2 = docs[half:]
        pool = multiprocessing.Pool(processes=2)
        results = pool.map(emotion_recognition, [docs1, docs2])
        pool.close()
        pool.join()
        results = results[0] + results[1]

        for i, doc in enumerate(docs):
            result = results[i]

            emotions = result['emotions']
            query = {'_id': doc['_id']}
            update = {'$set': {
                'processed': True,
                'error': False,
                'negative': emotions['مثبت'],
                'positive': emotions['منفی'],
                'unknown': emotions['نامشخص'],
                'very_new': True,
            }}
            collection.update_one(query, update)
        del docs, docs1, docs2, results