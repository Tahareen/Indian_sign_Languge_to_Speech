function speak() {
    // Call the server endpoint (which uses pyttsx3)
    fetch('/speak');
    
    // Also try to use browser TTS
    fetch('/get_text')
        .then(response => response.text())
        .then(text => {
            if(text) {
                const utterance = new SpeechSynthesisUtterance(text);
                window.speechSynthesis.speak(utterance);
            }
        })
        .catch(error => console.error('Error with browser speech:', error));
}