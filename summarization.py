from flask import Flask, render_template, request
import pickle
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from transformers import pipeline
summarization = Flask(__name__)

# Load the pre-trained T5 model and tokenizer
with open('t5_model.pkl', 'rb') as model_file:
    T5_model = pickle.load(model_file)
T5_PATH = 'google/flan-t5-base'
GPT_tokenier = T5Tokenizer.from_pretrained(T5_PATH)
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt") 
@summarization.route('/')
def home():
    return render_template('summrize.html')

@summarization.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        article = request.form['article']

        input_text = f"Summarized: {article}"
        summary1 = summarizer(input_text, max_length=150, min_length=30, do_sample=False)
        summary1 = summary1[0]['summary_text']
        input_ids = GPT_tokenier.encode(summary1, return_tensors="pt", max_length=1500,truncation=True)
        summary_ids = T5_model.generate(input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True, repetition_penalty=2.0, temperature=0.8 )
        gpt_summary = GPT_tokenier.decode(summary_ids[0], skip_special_tokens=True, max_length=1500)
        return render_template('summrize.html', article=article, gpt_summary= gpt_summary)
 
if __name__ == '__main__':
    summarization.run(debug=True)