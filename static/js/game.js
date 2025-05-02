document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const startButton = document.getElementById('start-game');
    const instructions = document.getElementById('instructions');
    const gamePlay = document.getElementById('game-play');
    const guessArea = document.getElementById('guess-area');
    const signImage = document.getElementById('sign-image');
    const guessPrompt = document.getElementById('guess-prompt');
    const progressBar = document.getElementById('progress-bar');
    const userGuess = document.getElementById('user-guess');
    const submitButton = document.getElementById('submit-guess');
    const replayButton = document.getElementById('replay-signs');
    const feedback = document.getElementById('feedback');
    const playAgainButton = document.getElementById('play-again');
    const newWordButton = document.getElementById('new-word');
    
    // Difficulty selector buttons
    const easyLevelBtn = document.getElementById('easy-level');
    const mediumLevelBtn = document.getElementById('medium-level');
    const hardLevelBtn = document.getElementById('hard-level');
    
    // Game state
    let currentWord = '';
    let currentIndex = 0;
    let imageInterval;
    let isShowingImages = false;
    let currentDifficulty = 'easy'; // Default difficulty
    
    // Event listeners for difficulty buttons
    easyLevelBtn.addEventListener('click', function() {
        setDifficulty('easy');
    });
    
    mediumLevelBtn.addEventListener('click', function() {
        setDifficulty('medium');
    });
    
    hardLevelBtn.addEventListener('click', function() {
        setDifficulty('hard');
    });
    
    function setDifficulty(difficulty) {
        currentDifficulty = difficulty;
        
        // Update active button
        [easyLevelBtn, mediumLevelBtn, hardLevelBtn].forEach(btn => {
            btn.classList.remove('active');
        });
        
        if (difficulty === 'easy') {
            easyLevelBtn.classList.add('active');
        } else if (difficulty === 'medium') {
            mediumLevelBtn.classList.add('active');
        } else if (difficulty === 'hard') {
            hardLevelBtn.classList.add('active');
        }
    }
    
    // Event listeners
    startButton.addEventListener('click', startGame);
    submitButton.addEventListener('click', checkGuess);
    replayButton.addEventListener('click', replayImages);
    playAgainButton.addEventListener('click', resetGame);
    newWordButton.addEventListener('click', getNewWord);
    
    // Start the game
    function startGame() {
        // Hide instructions, show game play area
        instructions.classList.add('hidden');
        gamePlay.classList.remove('hidden');
        
        getNewWord();
    }
    
    // Reset UI elements between games
    function resetUI() {
        userGuess.value = '';
        feedback.textContent = '';
        playAgainButton.classList.add('hidden');
        newWordButton.classList.add('hidden');
        submitButton.classList.remove('hidden');
        
        // Reset image and prompt states
        signImage.classList.remove('hidden');
        guessPrompt.classList.add('hidden');
        
        // Reset progress bar
        progressBar.style.width = '0%';
    }
    
    // Get a new word from the server
    function getNewWord() {
        resetUI();
        
        // If guessing area is visible, hide it and show game play area
        if (!guessArea.classList.contains('hidden')) {
            guessArea.classList.add('hidden');
            gamePlay.classList.remove('hidden');
        }
        
        // Fetch a random word from the server based on difficulty
        fetch(`/api/get-word?difficulty=${currentDifficulty}`)
            .then(response => response.json())
            .then(data => {
                currentWord = data.word;
                console.log("Word to guess:", currentWord); // For debugging
                startShowingImages();
            })
            .catch(error => {
                console.error('Error fetching word:', error);
            });
    }
    
    // Show images one by one
    function startShowingImages() {
        // Clear any existing interval
        if (imageInterval) {
            clearInterval(imageInterval);
        }
        
        // Ensure the image is visible and prompt is hidden
        signImage.classList.remove('hidden');
        guessPrompt.classList.add('hidden');
        
        currentIndex = 0;
        isShowingImages = true;
        
        // Show first image
        showNextImage();
        
        // Show each image for 2 seconds
        imageInterval = setInterval(() => {
            currentIndex++;
            
            if (currentIndex < currentWord.length) {
                // Show next letter's sign
                showNextImage();
            } else {
                // All images have been shown
                clearInterval(imageInterval);
                isShowingImages = false;
                
                // First hide the image
                signImage.classList.add('hidden');
                
                // Then show the "Guess it!!!" prompt
                guessPrompt.classList.remove('hidden');
                
                // After 2 seconds of "Guess it!!!" prompt, show guess area
                setTimeout(() => {
                    gamePlay.classList.add('hidden');
                    guessArea.classList.remove('hidden');
                    userGuess.focus();
                }, 2000);
            }
        }, 2000);
    }
    
    // Replay the images for the current word
    function replayImages() {
        // Don't allow replay if images are currently being shown
        if (isShowingImages) return;
        
        // Hide guess area and show game play area
        guessArea.classList.add('hidden');
        gamePlay.classList.remove('hidden');
        
        // Start showing images again (this will handle proper state reset)
        startShowingImages();
    }
    
    // Display the current letter's sign language image
    function showNextImage() {
        if (currentIndex < currentWord.length) {
            const letter = currentWord[currentIndex];
            signImage.src = `/static/images/${letter}.png`;
            
           
            signImage.alt = `Sign for letter ${letter}`;
            signImage.classList.remove('hidden');
            guessPrompt.classList.add('hidden');
            
            // Update progress bar
            const progress = ((currentIndex + 1) / currentWord.length) * 100;
            progressBar.style.width = `${progress}%`;
        }
        
    }
    
    // Check the user's guess
    function checkGuess() {
        const guess = userGuess.value.trim();
        
        if (guess === '') {
            feedback.textContent = 'Please enter a guess.';
            feedback.className = 'incorrect';
            return;
        }
        
        // Send the guess to the server for checking
        fetch('/api/check-answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                guess: guess,
                word: currentWord
            }),
        })
        .then(response => response.json())
        .then(data => {
            feedback.textContent = data.message;
            feedback.className = data.correct ? 'correct' : 'incorrect';
            
            if (data.correct) {
                // If correct, hide submit button and show play again button
                submitButton.classList.add('hidden');
                playAgainButton.classList.remove('hidden');
            } else {
                // If incorrect, show new word button
                newWordButton.classList.remove('hidden');
            }
        })
        .catch(error => {
            console.error('Error checking answer:', error);
        });
    }
    
    // Reset the game to play again with the same word
    function resetGame() {
        resetUI();
        replayImages();
    }
    
    // Allow pressing Enter to submit guess
    userGuess.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            checkGuess();
        }
    });
});