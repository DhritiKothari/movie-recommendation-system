# Movie Recommendation System

A content-based movie recommendation engine that suggests films based on similarity of content features like genres, plot, cast, and directors - no user ratings required!

## Features

- Content-Based Filtering: Recommends movies based on genres, plot summaries, keywords, main cast, and directors
- Interactive Interface: Demo mode with examples and interactive input for any movie title
- Smart Matching: Handles partial titles and suggests alternatives when exact matches aren't found
- Machine Learning: Uses TF-IDF vectorization and cosine similarity for accurate recommendations

##  Quick Start

### Prerequisites
- Python 3.7+
- TMDB dataset from Kaggle

### Installation

1. Clone the repository
   bash
   git clone <repository-url>
   cd movie-recommendation-system
   

2. Install dependencies
   bash
   pip install -r requirements.txt
   

3. Download dataset
   - Get the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle
   - Place both files (`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`) in the `data/` folder

4. Run the system
   bash
   python movie_recommender.py
  

## Usage

The system offers two modes:

1. Demonstration Mode: Shows recommendations for popular movies (The Dark Knight, Toy Story, etc.)
2. Interactive Mode: Type any movie title for personalized suggestions

Example queries to try:
- "The Dark Knight"
- "Toy Story" 
- "Inception"
- "The Avengers"

##  How It Works

 1. Data Processing
- **Handles missing values**: Automatically fills missing overviews, genres, keywords, and cast data
- **Text cleaning**: Converts JSON-like strings to Python objects using safe parsing
- **Feature extraction**: Extracts director, top 3 cast members, genres, and keywords from nested structures
- **Text normalization**: Lowercases and removes spaces for consistent processing

 2. Vectorization
- **TF-IDF Vectorization**: Uses scikit-learn's TfidfVectorizer with English stop words removal
- **Feature engineering**: Combines multiple features into a single text representation
- **Dimensionality control**: Limits features to 10,000 most important terms

 3. Similarity Calculation
- **Cosine Similarity**: Measures similarity between movie feature vectors
- **Similarity matrix**: Builds comprehensive matrix for all 5000+ movies
- **Ranking**: Sorts movies based on similarity scores

 4. Recommendation
- Returns top 5 most similar films based on content features
- Provides genre information, director, and plot overview

##  Technologies Used

- **Python** - Core programming language
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning (TF-IDF, cosine similarity)
- **NumPy** - Numerical computations

## Skills Demonstrated

Data Preprocessing
- Handling missing values for text and categorical data
- Converting JSON-like strings to structured Python objects
- Extracting specific features from nested data structures
- Text normalization and cleaning

 Vectorization & ML Techniques
- TF-IDF vectorization with stop word removal
- Feature engineering by combining multiple data sources
- Dimensionality control for efficient processing

 Similarity Measures
- Cosine similarity calculation
- Similarity matrix construction
- Ranking and sorting based on similarity scores

 ML/Recommendation Concepts
- Content-based filtering implementation
- Structured class-based design pattern
- Built-in evaluation and demonstration functionality

 Creative Implementation
- Multi-feature combination strategy
- Robust error handling and data validation
- Smart partial matching with user suggestions
- Interactive command-line interface
- Efficient processing of large datasets

## Contributing

Feel free to fork this project and submit pull requests for improvements:

- Add collaborative filtering capabilities
- Enhance the user interface with visualizations
- Extend with additional data sources or APIs
- Implement hybrid recommendation approaches
- Add rating prediction features



