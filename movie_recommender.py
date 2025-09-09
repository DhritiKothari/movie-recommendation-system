import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import warnings

warnings.filterwarnings("ignore")


def safe_literal_eval(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except (ValueError, SyntaxError):
        return []


def extract_director(crew):
    crew = safe_literal_eval(crew) if isinstance(crew, str) else crew
    if isinstance(crew, list):
        for person in crew:
            if isinstance(person, dict) and person.get("job") == "Director":
                return person.get("name", "")
    return ""


def extract_names(obj_list, key="name", top_n=None):
    if not isinstance(obj_list, list):
        return []
    names = [d.get(key, "") for d in obj_list if isinstance(d, dict)]
    return names[:top_n] if top_n else names


class MovieRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        self.feature_vectors = None
        self.cosine_sim = None

    def load_data(self):
        print("Loading datasets...")
        try:
            movies = pd.read_csv("data/tmdb_5000_movies.csv")
            credits = pd.read_csv("data/tmdb_5000_credits.csv")

            self.movies_df = movies.merge(
                credits, left_on="id", right_on="movie_id"
            )[
                ["id", "title_x", "overview", "genres", "keywords", "cast", "crew"]
            ]
            self.movies_df.rename(columns={"title_x": "title"}, inplace=True)

            print(f"Dataset loaded successfully with {len(self.movies_df)} movies")
            return True

        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            print("Please make sure the CSV files are in the 'data/' folder")
            return False

    def preprocess_data(self):
        print("Preprocessing data...")

        df = self.movies_df
        df.fillna({"overview": "", "genres": "[]", "keywords": "[]", "cast": "[]"}, inplace=True)

        for col in ["genres", "keywords", "cast", "crew"]:
            df[col] = df[col].apply(safe_literal_eval)

        df["director"] = df["crew"].apply(extract_director)
        df["top_cast"] = df["cast"].apply(lambda x: extract_names(x, top_n=3))
        df["genres_list"] = df["genres"].apply(extract_names)
        df["keywords_list"] = df["keywords"].apply(extract_names)

        def combine_features(row):
            features = (
                row["genres_list"]
                + row["keywords_list"]
                + row["top_cast"]
                + ([row["director"]] if row["director"] else [])
                + [row["overview"]]
            )
            return " ".join(str(f).lower().replace(" ", "") for f in features if f)

        df["combined_features"] = df.apply(combine_features, axis=1)
        print("Data preprocessing completed!")

    def build_model(self):
        print("Building recommendation model...")
        self.feature_vectors = self.vectorizer.fit_transform(
            self.movies_df["combined_features"]
        )
        self.cosine_sim = cosine_similarity(self.feature_vectors)
        print("Model built successfully!")

    def get_recommendations(self, movie_title, top_n=5):
        df = self.movies_df
        match = df[df["title"].str.lower() == movie_title.lower()]

        if match.empty:
            suggestions = df[df["title"].str.contains(movie_title, case=False)]
            if not suggestions.empty:
                print(f"Movie '{movie_title}' not found. Did you mean one of these?")
                for i, title in enumerate(suggestions["title"].head(3), 1):
                    print(f"{i}. {title}")
            else:
                print(f"Movie '{movie_title}' not found in the dataset.")
            return None

        idx = match.index[0]
        sim_scores = sorted(
            list(enumerate(self.cosine_sim[idx])),
            key=lambda x: x[1],
            reverse=True,
        )[1:top_n + 1]

        indices = [i for i, _ in sim_scores]
        return df.iloc[indices][["title", "genres_list", "overview", "director"]]

    def demonstrate_system(self):
        print("\n" + "=" * 60)
        print("MOVIE RECOMMENDATION SYSTEM DEMONSTRATION")
        print("=" * 60)

        examples = ["The Dark Knight", "Toy Story", "The Shawshank Redemption", "Inception"]
        
        for movie in examples:
            print(f"\nRecommendations based on '{movie}':")
            print("-" * 40)
            recs = self.get_recommendations(movie)
            if recs is not None:
                for i, (_, row) in enumerate(recs.iterrows(), 1):
                    print(f"{i}. {row['title']}")
                    print(f"   Genres: {', '.join(row['genres_list'][:3])}")
                    if row['director']:
                        print(f"   Director: {row['director']}")
                    if len(row['overview']) > 100:
                        print(f"   Overview: {row['overview'][:100]}...")
                    else:
                        print(f"   Overview: {row['overview']}")
                    print()

    def interactive_mode(self):
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)
        print("Type a movie title to get recommendations")
        print("Type 'exit' to quit the program")
        print("Type 'list' to see some available movies")

        while True:
            user_input = input("\nEnter movie title: ").strip()
            
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            elif user_input.lower() == "list":
                self.show_available_movies()
                continue
            elif not user_input:
                continue

            recs = self.get_recommendations(user_input)
            if recs is not None:
                print(f"\nTop 5 recommendations based on '{user_input}':")
                print("-" * 50)
                for i, (_, row) in enumerate(recs.iterrows(), 1):
                    print(f"{i}. {row['title']}")
                    print(f"   Genres: {', '.join(row['genres_list'][:3])}")
                    if row['director']:
                        print(f"   Director: {row['director']}")
                    print()

    def show_available_movies(self):
        print("\nSome available movies in the dataset:")
        print("-" * 40)
        sample_movies = self.movies_df['title'].sample(10).tolist()
        for i, movie in enumerate(sample_movies, 1):
            print(f"{i}. {movie}")


def main():
    print("Movie Recommendation System - Content Based Filtering")
    print("Initializing...")
    
    recommender = MovieRecommendationSystem()
    
    if recommender.load_data():
        recommender.preprocess_data()
        recommender.build_model()
        recommender.demonstrate_system()
        recommender.interactive_mode()


if __name__ == "__main__":
    main()
