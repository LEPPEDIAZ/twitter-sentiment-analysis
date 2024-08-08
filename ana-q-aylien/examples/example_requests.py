import requests
import json


def predict(text):
    data = {
        "text": text
    }
    headers = {'Content-type': 'application/json'}
    r = requests.post(
        'http://0.0.0.0:8000/predict',
        headers=headers,
        data=json.dumps(data)
    )
    print("Sentiment response:")
    print(r.text)



def main():
    predict("hello")


if __name__ == '__main__':
    main()
