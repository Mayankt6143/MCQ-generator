import streamlit as st
import pandas as pd
import random
from transformers import pipeline

# Set up the page title and layout
st.title("Automated MCQ Question Generator")
st.markdown("""
    This application generates multiple-choice questions (MCQs) from the input text using a transformer model.
    You can specify the number of questions and the number of options per question.
""")

# Load the GPT-2 model from Hugging Face's model hub (using st.cache_resource to cache the model)
@st.cache_resource
def load_model():
    return pipeline('text-generation', model='gpt2')

# Function to generate MCQs
def generate_mcqs(input_text, num_questions, num_options):
    model = load_model()
    mcqs = []

    for _ in range(num_questions):
        # Generate a text snippet as the basis for the question
        generated_text = model(input_text, max_length=1000, num_return_sequences=1)[0]['generated_text'].strip()
        
        # Create a question based on the input text
        question = f"What is the main idea of the text: '{input_text[:50]}...'"
        
        # Create a correct answer
        correct_option = generated_text[:30]
        
        # Generate random options that do not include the correct answer
        all_options = {correct_option}  # Start with the correct option
        while len(all_options) < num_options:
            # Generate random dummy options (could be refined further)
            dummy_option = f"Dummy option {random.randint(1, 1000)}"
            all_options.add(dummy_option)
        
        # Convert set to list and shuffle options
        all_options = list(all_options)
        random.shuffle(all_options)

        mcqs.append({
            'question': question,
            'options': all_options,
            'answer': correct_option  # Store the correct answer separately
        })

    return mcqs

# Create a text input box for users to enter their text (limit to 1000 characters)
input_text = st.text_area("Enter the text for MCQ generation:", "", max_chars=1000)
num_questions = st.number_input("Number of MCQs to generate:", min_value=1, max_value=10, value=1)
num_options = st.number_input("Number of options per MCQ:", min_value=2, max_value=5, value=4)

# Create a button to clear the inputs
if st.button("Clear Inputs"):
    st.experimental_rerun()  # This will reset the app state

# When the user clicks the button, generate the MCQs
if st.button("Generate MCQs"):
    if input_text:
        # Generate the MCQs
        mcqs = generate_mcqs(input_text, num_questions, num_options)
        
        # Display the generated questions and options
        for i, mcq in enumerate(mcqs):
            st.subheader(f"Generated MCQ {i + 1}:")
            st.write(f"**Question:** {mcq['question']}")
            
            st.write("**Options:**")
            for j, option in enumerate(mcq['options']):
                st.write(f"{chr(65 + j)}: {option}")

            st.write(f"**Correct Answer:** {mcq['answer']}")
            st.write("---")  # Separator for better readability
            
        # Provide a feedback mechanism
        correct_answers = [mcq['answer'] for mcq in mcqs]
        user_answers = []
        for i, mcq in enumerate(mcqs):
            user_answer = st.selectbox(f"Select the correct answer for MCQ {i + 1}:", options=mcq['options'], key=f'answer_{i}')
            user_answers.append(user_answer)
        
        if st.button("Check Answers"):
            for i, (correct, user) in enumerate(zip(correct_answers, user_answers)):
                if correct == user:
                    st.success(f"MCQ {i + 1}: Correct!")
                else:
                    st.error(f"MCQ {i + 1}: Incorrect! The correct answer is: {correct}")

        # Download functionality
        df = pd.DataFrame(mcqs)
        st.download_button("Download MCQs as CSV", df.to_csv(index=False).encode('utf-8'), "mcqs.csv", "text/csv")
    else:
        st.error("Please enter some text to generate MCQs.")
