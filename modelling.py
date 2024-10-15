import tkinter as tk
from tkinter import messagebox
from transformers import pipeline, AutoTokenizer
from collections import Counter

# Load the pre-trained model and tokenizer for argument mining and stance detection
model_name = "facebook/bart-large-mnli"
argument_mining_pipeline = pipeline("text-classification", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Backend function for argument mining and stance detection
def detect_argument_and_stance(text):
    results = []
    max_length = 512  # Maximum tokens the model can handle
    overlap = 50      # Number of overlapping tokens between chunks

    # Tokenize the text to get input ids
    tokenized_text = tokenizer.encode(text)
    
    # Process the input in chunks
    for i in range(0, len(tokenized_text), max_length - overlap):
        chunk = tokenized_text[i:i + max_length]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        try:
            # Use the NLP model to get results for each chunk
            chunk_results = argument_mining_pipeline(chunk_text)
            print(f"Chunk: {chunk_text}\nResults: {chunk_results}")  # Debug output
            results.extend(chunk_results)
        except Exception as e:
            return f"Error processing chunk: {str(e)}"
    
    return results

# Function to handle the analyze button click
def analyze_text():
    input_text = text_entry.get("1.0", tk.END).strip()  # Get text from input box
    if not input_text:
        messagebox.showerror("Input Error", "Please enter some text to analyze.")
        return

    # Call the backend function to analyze the text
    results = detect_argument_and_stance(input_text)

    # Aggregate results
    if results:
        label_counts = Counter(result['label'] for result in results)
        most_common_label = label_counts.most_common(1)
        aggregated_result = f"Most Common Label: {most_common_label[0][0]} (Count: {most_common_label[0][1]})\n"
    else:
        aggregated_result = "No results obtained."

    # Display the results in the output box
    output_text.delete("1.0", tk.END)  # Clear the previous output
    output_text.insert(tk.END, aggregated_result)

    # Optional: Display all individual results for debugging
    output_text.insert(tk.END, "Individual Results:\n")
    for result in results:
        label = result['label']
        score = result['score']
        output_text.insert(tk.END, f"{label}: {score:.4f}\n")

# Set up the main window using Tkinter
root = tk.Tk()
root.title("Argument Mining and Stance Detection")
root.geometry("600x400")

# Input label and text box for entering text
input_label = tk.Label(root, text="Enter Text for Analysis:")
input_label.pack()

text_entry = tk.Text(root, height=10, width=60)
text_entry.pack()

# Analyze button to trigger NLP analysis
analyze_button = tk.Button(root, text="Analyze Text", command=analyze_text)
analyze_button.pack()

# Output label and text box for showing the results
output_label = tk.Label(root, text="Analysis Result:")
output_label.pack()

output_text = tk.Text(root, height=10, width=60)
output_text.pack()

# Start the Tkinter main loop (runs the UI)
root.mainloop()
