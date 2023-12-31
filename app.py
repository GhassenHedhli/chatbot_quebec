from flask import Flask,render_template,jsonify,request

from chat import get_response

app = Flask(__name__, template_folder='template')
@app.get('/')
def index_get():
    return render_template('base.html')
@app.post('/predict')
def predict():
    text=request.get_json().get("message")
    response = get_response(text)
    message={"answer":response}
    return jsonify(message)
port_number = 3000

if __name__ == "__main__":
    app.run(debug=True,port=port_number)
