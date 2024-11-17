from flask import Flask, render_template, request, jsonify
import joblib
import pickle

# Load your trained model
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Load your trained model
#model = joblib.load('knn_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Collect data from form
    gre_verbal = float(request.form['gre_verbal'])
    gre_quant = float(request.form['gre_quant'])
    gre_analytical = float(request.form['gre_analytical'])
    gpa = float(request.form['gpa'])

    # Create a data point for prediction
    user_data = [[gre_verbal, gre_quant, gre_analytical, gpa]]
    recommended_university = model.predict(user_data)[0]

    return jsonify({'university': recommended_university})

if __name__ == '__main__':
    app.run(debug=True)
