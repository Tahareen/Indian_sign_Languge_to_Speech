function addSpace() {
    fetch('/add_space')
        .catch(error => console.error('Error adding space:', error));
}

function backspace() {
    fetch('/backspace')
        .catch(error => console.error('Error with backspace:', error));
}




function clearBuffer() {
    fetch('/clear')
        .catch(error => console.error('Error clearing buffer:', error));
}

function startCamera() {
    const videoElement = document.getElementById('video');
    
    // Call the /start endpoint to reset server state
    fetch('/start')
        .then(response => response.text())
        .then(data => {
            console.log(data); // Optional: Log the response
            // Set the video source to trigger the stream
            videoElement.src = "/video_feed";
            videoElement.style.display = "block";
        })
        .catch(error => console.error('Error starting camera:', error));
}

function stopStream() {
    const videoElement = document.getElementById('video');
    videoElement.src = ""; // Clear the video source to stop the stream
    videoElement.style.display = "none";
    
    fetch('/stop')
        .then(response => response.text())
        .then(data => {
            console.log(data); // Optional: Log the response
        })
        .catch(error => console.error('Error stopping stream:', error));
}
// Initialize speech synthesis variables
let speechSynthesis = window.speechSynthesis;
let availableVoices = [];
let preferredVoice = null;

// DOM elements
const speakButton = document.getElementById('speak-button');
const textDisplay = document.getElementById('text-display');
const statusElement = document.getElementById('status');

// Initialize voices when they become available
function setupVoices() {
    availableVoices = speechSynthesis.getVoices();
    
    // Try to find a good English voice
    preferredVoice = availableVoices.find(voice => 
        voice.lang === 'en-US' && voice.name.includes('Female')
    );
    
    // If no preferred voice found, just use the first English voice
    if (!preferredVoice) {
        preferredVoice = availableVoices.find(voice => 
            voice.lang.includes('en')
        );
    }
    
    // If still no voice, use the first available
    if (!preferredVoice && availableVoices.length > 0) {
        preferredVoice = availableVoices[0];
    }
    
    statusElement.textContent = `Ready (${availableVoices.length} voices available)`;
}

// Check if voices are immediately available or need to wait for the event
if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = setupVoices;
} else {
    setupVoices();
}

// Function to clean text before displaying or speaking
function cleanText(text) {
    // Ensure it's a string
    text = String(text);
    
    // If the text is comma-separated characters, join them
    if (text.includes(',') && text.split(',').every(char => char.length <= 1)) {
        text = text.split(',').join('');
    }
    
    return text;
}

// Function to speak text
function speakText(text) {
    // Cancel any ongoing speech
    speechSynthesis.cancel();
    
    // Clean the text
    text = cleanText(text);
    
    // Create a complete sentence to ensure proper pronunciation
    // Adding a period and surrounding with spaces helps for single words
    if (text.length > 0 && !/[.!?]$/.test(text)) {
        text = text + ".";
    }
    
    console.log("Speaking text:", text);
    
    // Create a new utterance
    const utterance = new SpeechSynthesisUtterance(text);
    
    // Explicitly set language to ensure proper word recognition
    utterance.lang = 'en-US';
    
    // Set voice if available
    if (preferredVoice) {
        utterance.voice = preferredVoice;
    }
    
    // Set properties
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    
    // Set up events
    utterance.onstart = function() {
        statusElement.textContent = "Speaking...";
        speakButton.disabled = true;
    };
    
    utterance.onend = function() {
        statusElement.textContent = "Finished speaking";
        speakButton.disabled = false;
    };
    
    utterance.onerror = function(event) {
        statusElement.textContent = `Error: ${event.error}`;
        speakButton.disabled = false;
    };
    
    // Speak the text
    speechSynthesis.speak(utterance);
}

// Function to fetch the text from Flask
function fetchAndSpeak() {
    statusElement.textContent = "Fetching text...";
    
    fetch('/get_text')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            let textToSpeak = data.text;
            
            // Clean the text before displaying and speaking
            textToSpeak = cleanText(textToSpeak);
            
            // Update the display with the clean text
            textDisplay.innerHTML = `<p>${textToSpeak}</p>`;
            
            // Speak the text
            speakText(textToSpeak);
        })
        .catch(error => {
            statusElement.textContent = `Error: ${error.message}`;
            console.error('There was a problem with the fetch operation:', error);
        });
}

// Add event listener to the speak button
speakButton.addEventListener('click', fetchAndSpeak);

// Initialize the page by fetching the text (but not speaking it)
fetch('/get_text')
    .then(response => response.json())
    .then(data => {
        const cleanedText = cleanText(data.text);
        textDisplay.innerHTML = `<p>${cleanedText}</p>`;
    })
    .catch(error => {
        console.error('Error fetching initial text:', error);
    });