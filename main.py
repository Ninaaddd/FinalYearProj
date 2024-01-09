# voice_assistant.py
import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import smtplib
import pyjokes
import pywhatkit
import pyowm
from newsapi import NewsApiClient


owm = pyowm.OWM('ea1c27935c886d9b6b465e2f23a44b0c')
newsapi = NewsApiClient(api_key='b8a263fc824049a692f4aed478f1bd96')

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def wishMe():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("Good Morning")
    elif 12 <= hour < 18:
        speak("Good Afternoon")
    else:
        speak("Good Evening")
    speak("How may I help you?")

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, languages=('en-in','hi-in','mar-in'))
        print(f"User Said: {query} \n")
    except Exception as e:
        print(e)
        speak("Can you please repeat")
        return "None"
    return query

def sendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    # Add your Gmail credentials
    server.login('yourgmail@gmail.com', 'your password')
    server.sendmail('tomail@gmail.com', to, content)
    server.close()
def get_weather(location):
    try:
        observation = owm.weather_manager().weather_at_place(location)
        weather = observation.weather
        temperature = weather.temperature('celsius')['temp']
        status = weather.detailed_status
        return f"The weather in {location} is {status} with a temperature of {temperature} degrees Celsius."
    except pyowm.exceptions.NotFoundError:
        return "Sorry, the city name was not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_latest_news():
    try:
        top_headlines = newsapi.get_top_headlines(language=('en','hi','mar'),country='ind')
        articles = top_headlines['articles']
        speak("Here are the latest news headlines.")
        for idx, article in enumerate(articles, start=1):
            title = article['title']
            descripton = article['description']
            print(f"Headline {idx}: {title}")
            print(f"Description: {descripton}")
            speak(f"Headline {idx}: {title}")
            speak(f"Description: {descripton}")
    except newsapi.exceptions.NewsAPIException as e:
        speak(f"Sorry, there was an issue fetching the news: {str(e)}")
    except Exception as e:
        speak(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    wishMe()
    query = takeCommand().lower()
    sites = [["youtube", "https://www.youtube.com"], ["wikipedia", "https://www.wikipedia.com"],
             ["google", "https://www.google.com"]]
    for site in sites:
        if f"Open {site[0]}".lower() in query:
            speak(f"Opening {site[0]}")
            webbrowser.open(site[1])

    if 'wikipedia' in query:
        speak('searching Wikipedia...')
        query = query.replace("wikipedia", "")
        results = wikipedia.summary(query, sentences=2)
        speak("According to Wikipedia")
        speak(results)

    elif 'time' in query:
        strTime = datetime.datetime.now().strftime("%H:%M:%S")
        speak(f"The time is {strTime}")

    elif 'date' in query:
        current_date = datetime.date.today()
        speak(f"The date is {current_date}")

    elif 'who is' in query:
        human = query.replace('who is', " ")
        info = wikipedia.summary(human, 2)
        print(info)
        speak(info)

    elif 'play' in query:
        song = query.replace('play', "")
        speak("playing" + song)
        pywhatkit.playonyt(song)

    elif 'joke' in query:
        speak(pyjokes.get_joke())

    elif 'weather' in query:
        speak("Sure, please specify the city name.")
        city = takeCommand().lower()
        weather_info = get_weather(city)
        speak(weather_info)
        print(weather_info)

    elif 'news' in query:
        get_latest_news()
