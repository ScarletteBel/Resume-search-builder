from flask import Flask, render_template, request
import Final_Project

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['input_data']
    results = Final_Project.pipeline(input_data, Final_Project.load_documents(source_dir = 'C:/Users/scarb/OneDrive/Documentos/BigData/GitHub/Resume-search-builder/Webapp/Resumes'))
    
    processed_results = ['\n'.join(map(str, res)) if isinstance(res, list) else str(res) for res in results]
    
    return render_template('index.html', results=processed_results)  # Pass results to the index.html template


if __name__ == '__main__':
    app.run(debug=True)