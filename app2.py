import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import random
import pandas as pd
import re
import os
from nltk.corpus import wordnet
import nltk
import PyPDF2  # For reading PDF files
import docx  # For reading Word files
from concurrent.futures import ThreadPoolExecutor  # For parallel processing
import requests  # For Wikipedia search
from bs4 import BeautifulSoup  # For HTML parsing

# Ensure WordNet is downloaded
nltk.download('wordnet', quiet=True)  # Download WordNet data quietly
nltk.download('punkt', quiet=True)  # Download Punkt tokenizer for sentence splitting

# Load the T5 model for question generation
@st.cache_resource
def load_t5_model():
    model_name = "valhalla/t5-base-qg-hl"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load the QA model for extracting correct answers from context
@st.cache_resource
def load_qa_model():
    return pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

# Function to clean input text
def clean_text(input_text):
    cleaned_text = re.sub(r'\s+', ' ', input_text.strip())
    return cleaned_text

# Function to split input text into chunks
def split_text(input_text, num_chunks):
    words = input_text.split()
    chunk_size = len(words) // num_chunks or 1  # Prevent division by zero
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)][:num_chunks]

# Function to generate logical distractors using WordNet
def generate_distractors(answer, num_distractors=3):
    distractors = set()
    
    # Find synonyms using WordNet
    for syn in wordnet.synsets(answer):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != answer.lower() and synonym not in distractors:
                distractors.add(synonym)
                if len(distractors) >= num_distractors:
                    break
        if len(distractors) >= num_distractors:
            break

    # If fewer distractors are found, add random words
    while len(distractors) < num_distractors:
        random_word = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))  # Random word
        if random_word != answer:
            distractors.add(random_word)

    return list(distractors)

# Function to search Wikipedia for a topic and return a specified number of words
def search_wikipedia(topic, word_limit):
    url = f"https://en.wikipedia.org/w/api.php?action=parse&page={topic.replace(' ', '_')}&prop=text&format=json"
    response = requests.get(url)

    if response.status_code == 200:
        page = response.json().get('parse')
        if page and 'text' in page:
            text = page['text']['*']
            # Use BeautifulSoup to remove HTML tags and clean up the text
            soup = BeautifulSoup(text, 'html.parser')
            clean_text = soup.get_text()
            words = clean_text.split()
            limited_text = ' '.join(words[:word_limit])  # Limit to the specified number of words
            
            # If the limited text is less than word_limit, log an informational message
            if len(words) < word_limit:
                st.warning(f"Retrieved only {len(words)} words from Wikipedia, which is less than your request.")
            
            return limited_text
    return None

# Function to generate MCQs in parallel
def generate_mcq(chunk, model, tokenizer, qa_model, num_distractors):
    question_prompt = f"generate question: {chunk}"
    inputs = tokenizer.encode(question_prompt, return_tensors='pt', max_length=512, truncation=True)

    # Generate the question
    output = model.generate(inputs, max_new_tokens=50)
    question = tokenizer.decode(output[0], skip_special_tokens=True)

    # Use the QA model to extract the correct answer
    answer = qa_model(question=question, context=chunk)['answer']

    # Generate logical distractors for the correct answer
    distractors = generate_distractors(answer, num_distractors)

    # Compile options, including the correct answer and distractors
    options = [answer] + distractors
    random.shuffle(options)  # Shuffle options to randomize the position of the correct answer
    
    return {'question': question, 'options': options, 'answer': answer}

# Function to handle parallel generation of MCQs
def generate_mcqs(input_text, num_questions, num_distractors):
    model, tokenizer = load_t5_model()
    qa_model = load_qa_model()
    
    chunks = split_text(input_text, num_questions)  # Split input text into chunks
    
    mcqs = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_mcq, chunk, model, tokenizer, qa_model, num_distractors) for chunk in chunks]
        for future in futures:
            mcqs.append(future.result())
    
    return mcqs

# Function to save the generated questions to a text file
def save_to_file(mcqs):
    df = pd.DataFrame(mcqs)
    file_path = 'generated_mcqs.txt'
    df.to_csv(file_path, sep='\t', index=False)
    return file_path

# Function to read text from a PDF file
def extract_text_from_pdf(uploaded_file, num_pages=5):  
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for i in range(min(num_pages, len(reader.pages))):
        text += reader.pages[i].extract_text()
    return text

# Function to read text from a Word document
def extract_text_from_word(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = "\n".join(para.text for para in doc.paragraphs)
    return text

# Streamlit UI setup
st.set_page_config(page_title="Automated MCQ Generator", layout="wide")
st.title("📝 Automated MCQ Question Generator")
st.markdown("Generate multiple-choice questions from your text input!")

# Tabs for Input (Text, File Upload, or Wikipedia Search)
tab1, tab2, tab3 = st.tabs(["Text Input", "File Upload", "Wikipedia Search"])

# Initialize summary variable
summary = ""

# Text Input Tab
with tab1:
    st.header("Enter Text for MCQ Generation")
    input_text = st.text_area("Enter the text for MCQ generation:", height=300)

# File Upload Tab
with tab2:
    st.header("Upload a PDF or Word Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]
        # Extract text based on file type
        if file_type == 'pdf':
            input_text = extract_text_from_pdf(uploaded_file, num_pages=3)  # Limit the number of pages processed
        elif file_type == 'docx':
            input_text = extract_text_from_word(uploaded_file)
        else:
            input_text = uploaded_file.read().decode("utf-8")  # For text files
        
        st.success("File uploaded successfully!")
        st.text_area("Extracted Text", value=input_text, height=300)

# Wikipedia Search Tab
with tab3:
    st.header("Search Wikipedia for a Topic")
    search_topic = st.text_input("Enter the topic you want to search for:")
    word_limit = st.number_input("Enter the number of words to extract from Wikipedia:", min_value=1, max_value=1000, value=100)

    if st.button("Search"):
        if search_topic:
            summary = search_wikipedia(search_topic, word_limit)  # Store summary in the variable
            if summary:
                st.success("Summary retrieved successfully!")
                st.session_state.summary = summary
                st.text_area("Retrieved Summary", value=summary, height=300)
            else:
                st.error("Could not retrieve summary from Wikipedia. Please check the topic.")

# Settings for MCQ Generation
with st.sidebar:
    st.header("Settings")
    num_questions = st.number_input("Number of MCQs to generate:", min_value=1, max_value=10, value=1)
    num_distractors = st.number_input("Number of distractors per question:", min_value=1, max_value=5, value=3)

    # Clear button
    if st.button("Clear All"):
        input_text = ""
        st.session_state.summary = ""  # Clear summary as well

# Main content area for generating MCQs
st.header("Generated MCQs")
if st.button("Generate MCQs"):
    # Use the summary from Wikipedia if the input text is empty
    text_to_use = input_text if input_text else st.session_state.get('summary', '')
    
    # Check the length of the text being used
    if text_to_use and len(text_to_use.split()) > 10:
        cleaned_text = clean_text(text_to_use)

        # Display progress bar
        progress = st.progress(0)

        # Generate MCQs in parallel
        mcqs = generate_mcqs(cleaned_text, num_questions, num_distractors)
        
        progress.progress(100)  # Complete the progress bar

        # Display the generated questions and options
        for i, mcq in enumerate(mcqs):
            st.subheader(f"Generated MCQ {i + 1}:")
            st.write(f"**Question:** {mcq['question']}")

            st.write("**Options:**")
            for idx, option in enumerate(mcq['options']):
                st.write(f"{chr(65 + idx)}: {option}")  # Display options as A, B, C, D
            
            st.write(f"**Answer:** {mcq['answer']}")
            st.write("---")  # Separator for better readability

        # Provide download link for the generated MCQs
        file_path = save_to_file(mcqs)
        st.download_button("Download MCQs", data=open(file_path, 'rb'), file_name=file_path, mime='text/csv')

    else:
        st.error("Please enter a valid text (at least 10 words) to generate MCQs.")

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("This application uses the T5 model for generating questions and a QA model for extracting answers. It is designed to help educators create MCQs quickly.")
