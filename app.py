from flask import Flask, render_template, jsonify, request, redirect, flash, session, url_for
from flask_pymongo import PyMongo
import subprocess
import os
from functools import wraps
import re
# from main2 import VoiceAssistant
# import main2


app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/voiceassistant"
app.secret_key = "shfgdekhh"

# mongodb://localhost:27017/
# mongodb://localhost:27017/

mongo = PyMongo(app)

# assistant = VoiceAssistant()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "is_user_logged_in" in session:
            return f(*args, **kwargs)
        else:
            flash("Unauthorized access", "warning")
            return redirect(url_for("login"))

    return decorated_function


@app.route('/', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        print("it's a post call")
        print(request.form["name"])
        name = request.form["name"]
        phnumber = request.form["phnumber"]

        # Validate that the phone number has exactly 10 digits
        if not re.match(r'^\d{10}$', phnumber):
            flash("Invalid phone number. Please enter exactly 10 digits.", "danger")
            return redirect(url_for('signup'))

        mongo.db.users.insert_one(
            {
                "name": name,
                "phnumber": phnumber,
            }
        )

        flash("User registered successfully", "success")

        return redirect(url_for('login'))

    print("it's a get call")
    return render_template('signup.html')


@app.route("/login/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        phnumber = request.form["phnumber"]
        found_user = mongo.db.users.find_one({"phnumber": phnumber})

        if not phnumber or not found_user:
            return render_template("login.html", error_message="It seems that this number is not registered. Please check again.")

        session["is_user_logged_in"] = True
        session["name"] = found_user["name"]
        session["phnumber"] = found_user["phnumber"]

        return redirect(url_for("index"))

    return render_template("login.html")


@app.route('/index/', methods=["GET", "POST"])
@login_required
def index():
    return render_template('index.html')


@app.route('/start-voice-assistant')
@login_required
def start_voice_assistant():
    try:
        # assistant.start()
        script_path = os.path.join(
            os.path.dirname(__file__), 'main2.py')
        subprocess.Popen(['python', script_path])
        return jsonify({'status': 'success', 'message': 'Voice assistant started successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route("/logout")
def logout():
    session.clear()
    flash("Successfully logged out", "success")
    return redirect(url_for("login"))


if __name__ == '__main__':
    # try:
    print(app.url_map)
    app.run(debug=True, port=5001)
    # finally:
    # assistant.stop()
