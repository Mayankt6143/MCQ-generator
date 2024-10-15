# app.py
import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import wikipedia
import os

# Load the essay generation model
essay_generator = pipeline("text-generation", model="gpt2")

# Set User-Agent for Wikipedia requests
wikipedia.set_lang("en")
wikipedia.set_user_agent("MyEssayGenerator/1.0 (myemail@example.com)")

# Initialize session state for clearing inputs
if 'topic' not in st.session_state:
    st.session_state.topic = ''
if 'google_results' not in st.session_state:
    st.session_state.google_results = []
if 'wiki_content' not in st.session_state:
    st.session_state.wiki_content = ''
if 'generated_text_google' not in st.session_state:
    st.session_state.generated_text_google = ''
if 'generated_text_wiki' not in st.session_state:
    st.session_state.generated_text_wiki = ''

# Function to get Wikipedia content
def get_wikipedia_content(topic):
    try:
        page = wikipedia.page(topic)
        return page.summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation error: {e.options}"
    except wikipedia.exceptions.PageError:
        return "Wikipedia page not found."

# Function to get Google search results
def get_google_search_results(topic):
    results = []
    for result in search(topic, num_results=5):  # Get the top 5 results
        results.append(result)
    return results

# Function to get content from a URL
def get_content_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text[:1000]  # Return the first 1000 characters for brevity
    except requests.RequestException as e:
        return f"Error fetching content: {e}"

# Function to limit words
def limit_words(text, max_words):
    words = text.split()
    return ' '.join(words[:max_words])

# Streamlit UI
st.title("Essay Generator with Search")
st.subheader("Choose your source and generate an essay!")

# User instructions
st.info("Enter a topic, choose a source, and generate an essay based on the content!")

# User input for the topic
st.session_state.topic = st.text_input("Enter a topic:", value=st.session_state.topic)

# Word limit input
word_limit = st.number_input("Set a maximum word limit for the essay (1-1000):", min_value=1, max_value=1000, value=100)

# Sidebar for options
st.sidebar.header("Options")
source_choice = st.sidebar.radio("Select a source:", ("Google", "Wikipedia"))

# Button to fetch Google search results
if source_choice == "Google" and st.button("Fetch Google Search Results"):
    if st.session_state.topic:
        with st.spinner("Fetching Google search results..."):
            st.session_state.google_results = get_google_search_results(st.session_state.topic)
        st.subheader("Google Search Results:")
        for idx, result in enumerate(st.session_state.google_results):
            st.write(f"{idx + 1}. {result}")

        # Let the user select a Google search result
        selected_google_index = st.number_input("Select a Google search result (1-5):", min_value=1, max_value=len(st.session_state.google_results), step=1)
        
        if st.button("Generate Essay from Google"):
            selected_url = st.session_state.google_results[selected_google_index - 1]
            with st.spinner("Generating essay from Google..."):
                google_content = get_content_from_url(selected_url)
                st.subheader("Content from Google:")
                st.write(google_content)

                # Generating essay based on the content
                prompt = f"Based on the following information, write an essay on the topic: {st.session_state.topic}\n\n{google_content}\n\n"
                # Truncate prompt to ensure it's within limits
                if len(prompt) > 2000:
                    prompt = prompt[:2000] 

                st.session_state.generated_text_google = essay_generator(
                    prompt,
                    max_new_tokens=300,
                    num_return_sequences=1,
                    pad_token_id=50256
                )[0]['generated_text']

                # Limit the generated text to the specified word limit
                limited_text = limit_words(st.session_state.generated_text_google, word_limit)

                # Display the generated essay
                st.subheader("Generated Essay from Google:")
                st.write(limited_text)

                # Option to download the generated essay
                st.download_button("Download Essay", data=limited_text, file_name="generated_essay_google.txt")

    else:
        st.warning("Please enter a topic.")

# Button to fetch Wikipedia content
if source_choice == "Wikipedia" and st.button("Fetch Wikipedia Content"):
    if st.session_state.topic:
        with st.spinner("Fetching Wikipedia content..."):
            st.session_state.wiki_content = get_wikipedia_content(st.session_state.topic)
        
        # Display the fetched content
        st.subheader("Content from Wikipedia:")
        st.write(st.session_state.wiki_content)

        # Generating essay based on the Wikipedia content
        prompt = f"Based on the following information, write an essay on the topic: {st.session_state.topic}\n\n{st.session_state.wiki_content}\n\n"
        # Truncate prompt to ensure it's within limits
        if len(prompt) > 2000:
            prompt = prompt[:2000]

        with st.spinner("Generating essay from Wikipedia..."):
            st.session_state.generated_text_wiki = essay_generator(
                prompt,
                max_new_tokens=300,
                num_return_sequences=1,
                pad_token_id=50256
            )[0]['generated_text']

        # Limit the generated text to the specified word limit
        limited_text = limit_words(st.session_state.generated_text_wiki, word_limit)

        # Display the generated essay
        st.subheader("Generated Essay from Wikipedia:")
        st.write(limited_text)

        # Option to download the generated essay
        st.download_button("Download Essay", data=limited_text, file_name="generated_essay_wikipedia.txt")

    else:
        st.warning("Please enter a topic.")

# Clear button to reset inputs and outputs
if st.button("Clear"):
    st.session_state.topic = ''
    st.session_state.google_results = []
    st.session_state.wiki_content = ''
    st.session_state.generated_text_google = ''
    st.session_state.generated_text_wiki = ''
    st.experimental_rerun()  # Re-run the app to reset the state
