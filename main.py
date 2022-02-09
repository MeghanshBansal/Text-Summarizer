# Imports
import gradio as gr
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from textblob import TextBlob

stop_words = list(STOP_WORDS)
nlp = spacy.load('en_core_web_sm')

# Adding  "\n" to the puctuation list to remove it
punctuation = punctuation + "\n"


def summarise(text):
    # Creating the spacy object
    doc = nlp(text)

    # Counting the frequency of each word
    word_frequency = {}
    for word in doc:
        if word.text.lower() not in stop_words and word.text.lower() not in punctuation:
            if word.text.lower() not in word_frequency.keys():
                word_frequency[word.text.lower()] = 1
            else:
                word_frequency[word.text.lower()] += 1

    maxFrequency = max(word_frequency.values())

    # Normalise the importance of each word
    for word in word_frequency.keys():
        word_frequency[word] = word_frequency[word]/maxFrequency

    # Giving each sentence scores based on importance and words in it
    sent_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sentence in sent_tokens:
        for word in sentence:
            if word.text.lower() in word_frequency.keys():
                if sentence not in sentence_scores.keys():
                    sentence_scores[sentence] = word_frequency[word.text.lower()]
                else:
                    sentence_scores[sentence] += word_frequency[word.text.lower()]

    # Taking 30% of best describing sentences from the text
    summary_size = int(len(sent_tokens)*0.3)
    if summary_size == 0:
        summary_size = 1
    summary = nlargest(summary_size, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    final_summary = ' '.join(final_summary)

    # Also predicting the sentiment of the text
    def sentiment(text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity < 0:
            return 'Negetive'
        elif polarity == 0:
            return 'Neutral'
        else:
            return 'Positive'
    return final_summary, sentiment(text)


# Creating interface for our model
iface = gr.Interface(
    fn=summarise,
    inputs=gr.inputs.Textbox(lines=15, label="ORIGINAL TEXT"),
    outputs=[gr.outputs.Textbox(label="SUMMARY"),
             gr.outputs.Textbox(label="SENTIMENT")],
    title="Text Summariser",
    theme="dark-grass",
    allow_flagging='never',
    layout='vertical',
)
iface.launch(server_port=8000, debug=False)
