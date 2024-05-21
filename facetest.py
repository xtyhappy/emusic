def emotion(file):
    import requests
    API_URL = "https://api-inference.huggingface.co/models/trpakov/vit-face-expression"
    headers = {"Authorization": "Bearer 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"}

    def query(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json()

    output = query(file)

    return output[0]['label'], output[1]['label']


# emotion1, emotion2 = emotion(file)
#
# print(emotion1, emotion2)
