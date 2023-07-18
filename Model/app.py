from pdf_loader import load_btyes_io
from core import pipeline

def inference(query, files, embedding_type):

    # pdfReader = PyPDF2.PdfReader(files[0])
    # text = ''
    # for page in pdfReader.pages:
    #     text += page.extract_text()
    # st.write(text)

    results, _ = pipeline(query, load_btyes_io(files), embedding_type=embedding_type)
    prob_per_documents = {result['name']: result['similarity'] for result in results}
    return prob_per_documents

sample_files = [
    "documents/business.pdf",
    "documents/data_science.pdf",
]        

sample_job_descriptions = {
    "Software Engineer": """We are looking for a software engineer with experience in Python and web development. The ideal candidate should have a strong background in building scalable and robust applications. Knowledge of frameworks such as Flask and Django is a plus. Experience with front-end technologies like HTML, CSS, and JavaScript is desirable. The candidate should also have a good understanding of databases and SQL. Strong problem-solving and communication skills are required for this role.
    """,
    "Data Scientist": """We are seeking a data scientist with expertise in machine learning and statistical analysis. The candidate should have a solid understanding of data manipulation, feature engineering, and model development. Proficiency in Python and popular data science libraries such as NumPy, Pandas, and Scikit-learn is required. Experience with deep learning frameworks like TensorFlow or PyTorch is a plus. Strong analytical and problem-solving skills are essential for this position.
    """ }   