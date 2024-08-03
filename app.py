from flask import Flask, render_template, request, jsonify
from transformers import pipeline

# Flask 애플리케이션 초기화
app = Flask(__name__)

# Stable LM 2 1.6B 모델 로드
model_name = "stabilityai/stablelm-2-1_6b"
pipe = pipeline("text-generation", model=model_name)

def generate_text(prompt, max_new_tokens=50, temperature=0.7, do_sample=True, top_p=0.9):
    # 모델을 사용하여 텍스트 생성
    generated = pipe(
        prompt,
        max_length=len(prompt.split()) + max_new_tokens,  # prompt 길이 포함
        num_return_sequences=1,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample
    )
    return generated[0]['generated_text']

@app.route('/', methods=['GET', 'POST'])
def home():
    query = None
    response = None

    if request.method == 'POST':
        query = request.form['content']
        try:
            response = generate_text(query)
        except Exception as e:
            response = str(e)

    return render_template('index.html', query=query, response=response)

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_new_tokens = data.get('max_new_tokens', 50)
    temperature = data.get('temperature', 0.7)
    do_sample = data.get('do_sample', True)
    top_p = data.get('top_p', 0.9)

    try:
        generated_text = generate_text(prompt, max_new_tokens, temperature, do_sample, top_p)
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
