import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import smtplib
import pyjokes
import pywhatkit
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pyowm
from newsapi import NewsApiClient
import vertexai
from vertexai.language_models import TextGenerationModel

# Initialize API
owm = pyowm.OWM('ea1c27935c886d9b6b465e2f23a44b0c')
newsapi = NewsApiClient(api_key='b8a263fc824049a692f4aed478f1bd96')

# Initialize BERT tokenizer and model


model_path = "D:/Ninad/SeventhProject/intent_model.pth"
model_state_dict = torch.load(model_path)


class IntentModel(nn.Module):
    def __init__(self, bert_model):
        super(IntentModel, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits


model = IntentModel(
    BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6))
model.load_state_dict(model_state_dict)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("intent_model.pth", num_labels=6)  # Update num_labels based on your dataset
# model_state_dict = torch.load('intent_model.pth')
# model.load_state_dict(model_state_dict)

# Define RNN model


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# defining methods


def get_weather(location):
    try:
        location = str(location)
        observation = owm.weather_manager().weather_at_place(location)
        weather = observation.weather
        temperature = weather.temperature('celsius')['temp']
        status = weather.detailed_status
        return f"The weather in {location} is {status} with a temperature of {temperature} degrees Celsius."
    except pyowm.exceptions.api_response_error.NotFoundError:
        return "Sorry, the city name was not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def get_latest_news():
    try:
        top_headlines = newsapi.get_top_headlines(language='en', country='in')
        articles = top_headlines['articles']
        news_headlines = []
        for idx, article in enumerate(articles, start=1):
            title = article['title']
            description = article['description']
            news_headlines.append((title, description))
        return news_headlines
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


def get_search_results(query):
    try:
        vertexai.init(project="total-entity-409908", location="us-central1")
        parameters = {
            "candidate_count": 1,
            "max_output_tokens": 1024,
            "temperature": 0.9,
            "top_p": 1
        }
        model = TextGenerationModel.from_pretrained("text-bison")
        response = model.predict(
            query,
            **parameters
        )
        return f"{response.text}"
    except Exception as e:
        return f"An unexpected error occured: {str(e)}"


# Define voice assistant


class VoiceAssistant:
    def __init__(self):
        self.engine = pyttsx3.init('sapi5')
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[0].id)
        # Assuming input_size is the BERT output size
        self.rnn_model = RNNModel(
            input_size=768, hidden_size=128, output_size=7)  # Update output_size based on your dataset

    def speak(self, audio):
        self.engine.say(audio)
        self.engine.runAndWait()

    def take_command(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            r.pause_threshold = 1
            audio = r.listen(source)

        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language='en-in')
            print(f"User Said: {query} \n")
        except Exception as e:
            print(e)
            self.speak("Can you please repeat")
            return "None"
        return query

    def classify_intent(self, query):
        inputs = tokenizer(query, return_tensors="pt",
                           max_length=128, truncation=True)
        outputs = model(input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask)
        probs = F.softmax(outputs, dim=1)
        predicted_label = torch.argmax(probs).item()
        intents = ['time', 'weather', 'search', 'joke',
                   'news', 'wikipedia']  # Update with your intents
        return intents[predicted_label]

    def generate_response(self, intent, query):
        if intent == "time":
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            return f"The time is {strTime}"
        elif intent == "weather":
            location = self.speak("Sure, please specify the city name.")
            return get_weather(location)
        elif intent == "news":
            news = get_latest_news()
            if isinstance(news, list):
                response = "Here are the latest news headlines:\n"
                for idx, (title, description) in enumerate(news, start=1):
                    response += f"Headline {idx}: {title}\n"
                    response += f"Description: {description}\n"
                return response
            else:
                return news  # Handle error messages
        elif intent == "search":
            search_media = get_search_results(query)
            return search_media
        else:
            return "How may I help you?"

    def start(self):
        self.speak("Initializing voice assistant...")
        self.speak("Hello! I am your voice assistant.")
        while True:
            query = self.take_command().lower()
            intent = self.classify_intent(query)
            response = self.generate_response(intent, query)
            self.speak(response)


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.start()
