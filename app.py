from flask import Flask, jsonify, request, url_for, render_template

app = Flask(__name__)

@app.route('/static/<path:path>')
def serve_static(path):
    return url_for('static', filename=path)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
