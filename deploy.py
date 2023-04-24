from flask import Flask, request, jsonify
import joblib
import pandas as pd

model = joblib.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from user
    data = request.get_json(force=True)

    # Extract features from input data
    data = pd.DataFrame(data)
    data['Month'] = data['Month'].map({'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12})
    data['VisitorType'] = data['VisitorType'].map({'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 0})
    data['Weekend'] = data['Weekend'].map({True: 1, False: 0})
    numeric_features = ['Administrative', 'Informational', 'ProductRelated', 'ExitRates', 'PageValues']

    for feature in numeric_features:
        data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()
    data['TotalTimeOnSite'] = data['Administrative_Duration'] + data['Informational_Duration'] + data['ProductRelated_Duration']
    data['TotalPageviews'] = data['Administrative'] + data['Informational'] + data['ProductRelated']
    
    # Make prediction using the trained model
    res = {}
    if data.shape[0]==1:
        label = model.predict(data)[0]
        prob = model.predict_proba(data)[0][label]
        res['prediction_0'] = {'label':int(label), 'score':float(prob)}
    else:
        labels = model.predict(data)
        probs = model.predict_proba(data)
        for i in range(len(labels)):
            label = int(labels[i])
            prob = float(probs[i][labels[i]])
            res['prediction_{}'.format(i)] = {'label':label, 'score':prob}
            

    # Return prediction result
    return jsonify(res)

if __name__ == '__main__':
    app.run(port=5000, debug=False)