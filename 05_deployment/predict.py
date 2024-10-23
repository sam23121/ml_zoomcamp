import pickle

from flask import Flask
from flask import request
from flask import jsonify


# model_file = 'model1.bin'
# dv_file = 'dv.bin'
with open('model2.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('default')

@app.route('/predict', methods=['POST'])
def predict():
    loan = request.get_json()

    X = dv.transform([loan])
    y_pred = model.predict_proba(X)[0, 1]
    default = y_pred >= 0.5

    result = {
        'default_probability': float(y_pred),
        'default': bool(default)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)