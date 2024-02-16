from flask import Flask, jsonify, send_from_directory
import os

app = Flask(__name__)

frontend_dir = os.path.join(os.path.dirname(__file__), 'Frontend')

@app.route('/')
def index():
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Sample results
    result = "Normal"
    confidence = 90
    return jsonify(result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)