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
import vertexai
from vertexai.language_models import TextGenerationModel

owm = pyowm.OWM('ea1c27935c886d9b6b465e2f23a44b0c')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "total-entity-409908-e2559bcbedd7.json"

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
    observation = owm.weather_at_place(location)
    weather = observation.get_weather()
    temperature = weather.get_temperature()
    status = weather.get_status()
    return  f"The weather in {location} is {status} with a temperature of {temperature} degree Celsius"

def get_vertex(text):
    vertexai.init(project="total-entity-409908", location="us-central1")
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0.9,
        "top_p": 1
    }

    # Check if text is available
    if text:
        model = TextGenerationModel.from_pretrained("text-bison")
        response = model.predict(text, **parameters)
        return response
    else:
        speak("No input available for generation.")

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

    else:
        response = get_vertex(query)
        speak(response)