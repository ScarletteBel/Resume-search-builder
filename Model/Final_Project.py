# !pip install nltk==3.8.1
# !pip install numpy==1.24.3
# !pip install PyPDF2==3.0.1
# !pip install scikit_learn==1.2.2
# !pip install sentence_transformers==2.2.2
# !pip install fastapi
# !pip install uvicorn[standard]
# !pip install python-multipart
# !pip install python-dotenv


import os
import PyPDF2
import re
import unicodedata
import nltk
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def load_single_document(file_path: str):
    # Loads a single document from file path
    if file_path[-4:] == '.txt':
        with open(file_path, 'r') as f:
            return f.read()

    elif file_path[-4:] == '.pdf':
        pdfFileObj = open(file_path, 'rb')
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        text = ''
        for page in pdfReader.pages:
            text += page.extract_text()
        return text

    elif file_path[-4:] == '.csv':
        with open(file_path, 'r') as f:
            return f.read()

    else:
        raise Exception('Invalid file type')


def load_documents(file_paths: list[str] = None, source_dir: str = None):
    # Loads all documents from source documents directory
    if file_paths:
        all_files = file_paths
    elif source_dir:
        all_files = [os.path.abspath(os.path.join(source_dir, file)) for file in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, file))]
    else:
        raise Exception('No file paths or source directory provided')

    return [
            {
                'name': os.path.basename(file_path),
                'content': load_single_document(f"{file_path}")
            } for idx, file_path in enumerate(all_files) if file_path[-4:] in ['.txt', '.pdf', '.csv']
        ]

def load_io(file_byte = None):
    # Loads a single document from file path
    if file_byte.name[-3:] == 'txt':
        return file_byte.read().decode("utf-8")

    elif file_byte.name[-3:] == 'pdf':
        pdfReader = PyPDF2.PdfReader(file_byte)
        text = ''
        for page in pdfReader.pages:
            text += page.extract_text()
        return text

    else:
        raise Exception('Invalid file type')

def load_btyes_io(files = None):

    return [
        {
            'name': file_btye.name,
            'content': load_io(file_btye)
        } for idx, file_btye in enumerate(files) if file_btye.name[-3:] in ['txt', 'pdf']
    ]

def embedding(documents, embedding='bert'):
    if embedding == 'bert':
        sbert_model = SentenceTransformer('bert-base-nli-mean-tokens', cache_folder=os.path.join(os.getcwd(), 'embedding'))

        document_embeddings = sbert_model.encode(documents)
        return document_embeddings

    if embedding == 'minilm':
        sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=os.path.join(os.getcwd(), 'embedding'))

        document_embeddings = sbert_model.encode(documents)
        return document_embeddings

    if embedding == 'tfidf':
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True, stop_words='english')
        word_vectorizer.fit(documents)
        word_features = word_vectorizer.transform(documents)

        return word_features


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words



def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        # print(word)
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words


def preprocess(documents):
    preprocessed_documents = []
    for document in documents:
        tokens = nltk.word_tokenize(document)
        preprocessed = normalize(tokens)
        preprocessed = ' '.join(map(str, preprocessed))
        preprocessed_documents.append(preprocessed)

    return preprocessed_documents

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def pipeline(input_doc:str , ori_documents, embedding_type='bert'):
    documents = np.array([doc['content'] for doc in ori_documents])
    documents = np.insert(documents, 0, input_doc)
    print(documents)
    preprocessed_documents = preprocess(documents)
    #print(preprocessed_documents)
    print("Encoding with BERT...")
    documents_vectors = embedding(preprocessed_documents, embedding=embedding_type)
    #print(documents_vectors)
    print("Encoding finished")

    #compute cosine similarity
    pairwise = cosine_similarity(documents_vectors)
    #print(pairwise)
    #only retain useful information
    pairwise = pairwise[0,1:]
    sorted_idx = np.argsort(pairwise)[::-1]
    result_pairwise = pairwise[sorted_idx]

    results = []
    print('Resume ranking:')
    for idx in sorted_idx:
        single_result = {
            'rank': idx,
            'name': ori_documents[idx]['name'],
            'similarity': pairwise[idx].item()
        }
        results.append(single_result)
        print(f'Resume of candidite {idx}')
        print(f'Cosine Similarity: {pairwise[idx]}\n')

    return results, result_pairwise



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
    "/content/resumes/Resume_Fernando_Hinojosa.pdf",]

sample_job_descriptions = {
    "Software Engineer": """We are looking for a software engineer with experience in Python and web development. The ideal candidate should have a strong background in building scalable and robust applications. Knowledge of frameworks such as Flask and Django is a plus. Experience with front-end technologies like HTML, CSS, and JavaScript is desirable. The candidate should also have a good understanding of databases and SQL. Strong problem-solving and communication skills are required for this role.
    """,
    "Data Scientist": """We are seeking a data scientist with expertise in machine learning and statistical analysis. The candidate should have a solid understanding of data manipulation, feature engineering, and model development. Proficiency in Python and popular data science libraries such as NumPy, Pandas, and Scikit-learn is required. Experience with deep learning frameworks like TensorFlow or PyTorch is a plus. Strong analytical and problem-solving skills are essential for this position.
    """}



if __name__ == '__main__':
    pipeline('''About Sleek

Sleek is on a mission to revolutionize how entrepreneurs operate their business. We want to give small business owners peace of mind and the power of online solutions to allow them to focus on what they do best - growing their business. As we work for our thousands of customers, we gather millions of data points about their business, and in turn we transform those into useful, actionable insights and recommendations to accelerate their growth through smart algorithms.

We are a team of 400 builders from 17 countries, with offices in Singapore, Philippines, Hong Kong, Australia and the UK committed to delivering a delightful experience to our clients!

You will be working in the Data & Analytics organization to solve a wide range of business problems leveraging advanced analytics. You will deploy a flexible analytical skill set to deliver insightful data and analysis and model business scenarios. Your principal goal will be to use data to drive better business decisions. This means translating data into meaningful insights and recommendations and, where relevant, proactively implement improvements. You will be developing the business reporting and analysis for our internal operations world-wide. The job will require working closely with the various Business Units to understand their business question as well as the whole data team to understand and access available data.

Position Duties
Drive analytical problem-solving and deep dives. Work with large, complex data sets. Solve difficult, non-routine problems, applying advanced quantitative methods.
Collaborate with a wide variety of cross-functional partners to determine business needs, drive analytical projects from start to finish.
Align with involved stakeholders to set up dashboards and reports to drive data driven decision across all departments
Working very closely with our Data team, Tech and Product team to understand the business logic to generate accurate reports and correct analysis

Requirements
Data Analysis
Performance Standards
Able to commit for a period of at least 4 months
Currently pursuing a degree in Business Science, Engineering or relevant disciplines with a focus on data.
Good knowledge in SQL, R and Python.
Experience in data visualization tools (Tableau, PowerBI, Google DataStudio or equivalent) will be an added advantage.''',
                   load_documents(source_dir = '/content/resumes'))



