from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logger
import numpy as np
import sys
import json

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-digikala")
model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-digikala")

def softmax(inputs, all):
    all_ = [float(all[0]), float(all[1]), float(all[2])]
    return np.exp(float(inputs)) / np.sum(np.exp(all_))

def emotion_recognition(texts):
    results = []
    for text in texts:
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
        results.append(result)

    return results

# emotions = emotion_recognition(['جای بانک‌ها بابورس عوض شده  برای تورم'])
# print (emotions)

# read input from stdin

try:
    # read input from stdin
    for line in sys.stdin:
        data = json.loads(line)
        texts = data['texts']

        # call emotion_recognition function
        results = emotion_recognition(texts)

        # send result to stdout
        print(json.dumps(results))
        sys.stdout.flush()
except BrokenPipeError:
    pass