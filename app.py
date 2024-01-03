# app.py
from flask import Flask, render_template, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-voice-assistant')
def start_voice_assistant():
    try:
        # Assuming voice_assistant.py is in the same directory
        script_path = os.path.join(os.path.dirname(__file__), 'main.py')
        subprocess.Popen(['python', script_path])
        return jsonify({'status': 'success', 'message': 'Voice assistant started successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
