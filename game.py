import random
from flask import render_template, request, jsonify

# Lists of words for different difficulty levels
EASY_WORDS = [
    "CAT", "DOG", "SUN", "HAT", "BIG", "RUN", "FUN", "JAM", "ZIP", "BOX", 
    "FOX", "DAY", "MAP", "PEN", "RED", "SKY", "TOP", "WIN", "YES", "NOW"
]

MEDIUM_WORDS = [
    "APPLE", "BEACH", "CLOUD", "DANCE", "EARTH", "FLAME", "GRAPE", "HOUSE",
    "JUICE", "KITE", "LEMON", "MONEY", "NORTH", "OCEAN", "PLANT", "QUEEN",
    "RIVER", "SMILE", "TIGER", "WATER"
]

HARD_WORDS = [
    "DOLPHIN", "ELEPHANT", "FANTASY", "GRADUATE", "HOSPITAL", "INTERNET",
    "JOURNEY", "KNOWLEDGE", "LAUGHTER", "MOUNTAIN", "NOTEBOOK", "PAINTING",
    "QUESTION", "RAINFALL", "SOLUTION", "TOMORROW", "UNIVERSE", "VACATION",
    "WILDLIFE", "YEARBOOK"
]

def get_game_routes(app):
    @app.route('/game')
    def game():
        """Render the game page"""
        return render_template('game.html')
    
    @app.route('/api/get-word')
    def get_random_word():
        """API endpoint to get a random word for the game"""
        difficulty = request.args.get('difficulty', 'easy')
        
        if difficulty == 'easy':
            word = random.choice(EASY_WORDS)
        elif difficulty == 'medium':
            word = random.choice(MEDIUM_WORDS)
        elif difficulty == 'hard':
            word = random.choice(HARD_WORDS)
        else:
            word = random.choice(EASY_WORDS)  # Default to easy
            
        return jsonify({"word": word})
    
    @app.route('/api/check-answer', methods=['POST'])
    def check_answer():
        """API endpoint to check if the guess is correct"""
        data = request.get_json()
        user_guess = data.get('guess', '').strip().upper()
        correct_word = data.get('word', '').strip().upper()
        
        is_correct = user_guess == correct_word
        
        return jsonify({
            "correct": is_correct,
            "message": "Good job!" if is_correct else "Try again!"
        })