🎬 Movie Recommendation System
A smart content-based movie recommendation engine that suggests films based on what makes movies similar - no user ratings required!

✨ Features
Content-Based Filtering: Recommends movies based on genres, plot, cast, directors, and keywords
Interactive Interface: Demo mode and interactive input for any movie title
Smart Matching: Handles partial titles and suggests alternatives
Machine Learning: Uses TF-IDF vectorization and cosine similarity

🚀 Quick Start
Prerequisites
Python 3.7+
TMDB dataset from Kaggle

Installation
Clone the repository
bash
git clone <your-repo-url>
cd movie-recommendation-system
Install dependencies
bash
pip install -r requirements.txt

Download dataset
Get TMDB 5000 Movie Dataset
Place tmdb_5000_movies.csv and tmdb_5000_credits.csv in the data/ folder

Run the system
bash
python movie_recommender.py
📁 Project Structure
text
movie-recommendation-system/
├── data/                 # Dataset files
├── movie_recommender.py  # Main application
├── requirements.txt      # Dependencies
└── README.md            # This file
🎯 Usage
The system offers two modes:
Demonstration Mode: Shows recommendations for popular movies
Interactive Mode: Type any movie title for personalized suggestions

Example queries:
"The Dark Knight"
"Toy Story"
"Inception"
"The Avengers"

🔧 How It Works
Data Processing: Combines genres, keywords, cast, directors, and overview
Vectorization: Converts text to numerical features using TF-IDF
Similarity Calculation: Uses cosine similarity to find matching movies
Recommendation: Returns top 5 most similar films

📊 Technologies Used
Python - Core programming
pandas - Data manipulation
scikit-learn - Machine learning (TF-IDF, cosine similarity)
NumPy - Numerical computations

🤝 Contributing
Feel free to fork this project and submit pull requests for improvements:
Add collaborative filtering
Enhance the UI
Extend with additional data sources

📝 License
Educational project using TMDB dataset from Kaggle.

