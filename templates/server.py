from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

openai.api_key = 'sk-proj-gp8KSIANNi46OpmObHtGT3BlbkFJJEvJLookKLqkgNsqi1Kl'

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data['prompt']

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a trading bot analyzing cryptocurrency markets."},
            {"role": "user", "content": prompt}
        ]
    )

    return jsonify(response.choices[0].message['content'].strip())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
