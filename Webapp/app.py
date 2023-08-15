from flask import Flask, render_template, request, url_for
import Final_Project
import ast 

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['input_data']
    results = Final_Project.pipeline(input_data, Final_Project.load_documents(source_dir = 'C:/Users/scarb/OneDrive/Documentos/BigData/GitHub/Resume-search-builder/Webapp/static/Resumes'))

    formatted_details = []
    formatted_scores = []

    if results and len(results) >= 2:
        details = results[0]  
        if isinstance(details, str):  # Check if the result is a string representation of a list
            details = ast.literal_eval(details)
        formatted_details = [
            {
                "rank": d['rank'],
                "link": url_for('static', filename='Resumes/' + d['name']),
                "name": d['name'],
                "similarity": d['similarity']
            } for d in details
        ]

        # Second element seems to be the scores
        scores = results[1]
        if isinstance(scores, str):  # Check if the result is a string representation of a list
            scores = ast.literal_eval(scores)
        formatted_scores = ' '.join(map(str, scores))

        
    else:
        error_message = "Error: Unexpected results format"
        return render_template('index.html', error=error_message)

    
    return render_template('index.html', details=formatted_details, scores=formatted_scores)


if __name__ == '__main__':
    app.run(debug=True)