from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Model names
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
summarization_model_name = "facebook/bart-large-cnn"
reply_generation_model_name = "google/flan-t5-large"

# Load and save models with tokenizers
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)

torch.save(sentiment_model.state_dict(), "sentiment_model.pth")
sentiment_tokenizer.save_pretrained("sentiment_tokenizer")

summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)

torch.save(summarization_model.state_dict(), "summarization_model.pth")
summarization_tokenizer.save_pretrained("summarization_tokenizer")

reply_model = AutoModelForSeq2SeqLM.from_pretrained(reply_generation_model_name)
reply_tokenizer = AutoTokenizer.from_pretrained(reply_generation_model_name)

torch.save(reply_model.state_dict(), "reply_generation_model.pth")
reply_tokenizer.save_pretrained("reply_tokenizer")

# Functions to load models
def load_sentiment_model():
    model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    tokenizer = AutoTokenizer.from_pretrained("sentiment_tokenizer")
    model.load_state_dict(torch.load("sentiment_model.pth", map_location="cpu"))
    return model, tokenizer

def load_summarization_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
    tokenizer = AutoTokenizer.from_pretrained("summarization_tokenizer")
    model.load_state_dict(torch.load("summarization_model.pth", map_location="cpu"))
    return model, tokenizer

def load_reply_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(reply_generation_model_name)
    tokenizer = AutoTokenizer.from_pretrained("reply_tokenizer")
    model.load_state_dict(torch.load("reply_generation_model.pth", map_location="cpu"))
    return model, tokenizer

# NLP Functions
def analyze_sentiment(text):
    """Perform sentiment analysis using saved model."""
    model, tokenizer = load_sentiment_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)

    # Get sentiment prediction
    label = outputs.logits.argmax().item()

    # Map the correct label based on nlptown's model (1-5 star ratings)
    sentiment_labels = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive"
    }

    return {
        "label": sentiment_labels.get(label, "Unknown"),
        "score": outputs.logits.max().item()
    }


def summarize_text(text):
    model, tokenizer = load_summarization_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(**inputs, max_length=100, min_length=30, num_beams=5, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_reply(email_body):
    model, tokenizer = load_reply_model()
    inputs = tokenizer(email_body, return_tensors="pt", truncation=True, max_length=1024)
    reply_ids = model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
    return tokenizer.decode(reply_ids[0], skip_special_tokens=True)

def process_email_content(email_body):
    sentiment = analyze_sentiment(email_body)
    summary = summarize_text(email_body)
    reply = generate_reply(email_body)
    
    return {
        "sentiment": sentiment,
        "summary": summary,
        "reply": reply
    }
