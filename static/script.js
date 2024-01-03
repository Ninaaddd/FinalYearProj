
const buttons = document.querySelectorAll(".hover-button");
const synth = window.speechSynthesis;

buttons.forEach(button => {
    button.addEventListener("mouseenter", () => {
        const text = button.textContent;
        const utterance = new SpeechSynthesisUtterance(text);
        synth.speak(utterance);
    });
});



const getBatteryButton = document.getElementById("get-battery");
const batteryPercentageDisplay = document.getElementById("battery-percentage");

// Check if the Battery Status API is supported
if ("getBattery" in navigator && "speechSynthesis" in window) {
    getBatteryButton.addEventListener("click", async () => {
        try {
            const battery = await navigator.getBattery();
            const percentage = (battery.level * 100).toFixed(2);
            batteryPercentageDisplay.textContent = `Battery Percentage: ${percentage}%`;

            // Speak the battery percentage
            const utterance = new SpeechSynthesisUtterance(`Battery Percentage: ${percentage} percent.`);
            speechSynthesis.speak(utterance);
        } catch (error) {
            batteryPercentageDisplay.textContent = "Battery information not available.";
            console.error(error);
        }
    });
} else {
    // If the Battery Status API or Speech Synthesis API is not supported, display an error message
    getBatteryButton.disabled = true;
    getBatteryButton.textContent = "Battery API or Speech Synthesis Not Supported";
}

// script.js

document.addEventListener("DOMContentLoaded", function () {
    var askMeButton = document.getElementById("ask-me-button");

    askMeButton.addEventListener("click", function () {
        fetch('/start-voice-assistant')
            .then(response => response.json())
            .then(data => console.log(data))
            .catch(error => console.error('Error:', error));
    });
});


    