from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Define a function to create a question answering pipeline
def create_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")

# Create the pipeline outside of the request handling for efficiency
qa_pipeline = create_qa_pipeline()

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle user query
@app.route('/answer', methods=['POST'])
def answer_question():
    # Get user input from form
    question = request.form['question']
    context = request.form['context']

    # Use the QA pipeline to get an answer
    answer = qa_pipeline({'question': question, 'context': context})

    # Render the answer page with the result
    return render_template('index.html', question=question, context=context, answer=answer['answer'])

if __name__ == '__main__':
    app.run(debug=True)
