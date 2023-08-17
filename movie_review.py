import pandas as pd
import numpy as np

# Sample movie review dataset
data = {
    'movie': [
        "The Lord of the Rings The Two Towers",
        "Inception",
        "Spiderman Across the spider verse",
        "The Dark Knight",
        "Three colors red",
        "It happened one night",
        "In the Mood for Love",
        "Before Sunrise",
        "Gone with the wind",
        "Eternal Sunshine of the Spotless Mind",
        "The Shawshank Redemption",
        "Raging Bull",
        "Lawrence of Arabia"
    ],
    'review': [
        "remarkable display of fantasy action powerful ring hobbit destroy it fight",
        "implanting stealing idea destroy gripping action jaw dropping fight stunning visual violence",
        "mind bending wild action sequences intimate emotional moments amazing action",
        "Best live action portrayal beat organized crime in Gotham enigmatic villain brutality violence",
        "mesmerising friendship turned love profound unconventional bond heartfelt",
        "Romantic comedy screwball comedy enduring tale of romance comical true love",
        "Neighbors solace bonding affair predicament spell binding infatuation heartwarming",
        "blossoming love know each other chance encounter meeting someone special fleeting romance magical evening",
        "epic romance greatest romantic film ever made touching amazing relationship cherished love",
        "Length people go finding love of life emotional rollercoaster in blossoming of love reignited fascinating journey into heart",
        "movie about friendship life fight to be good person prisoner surviving",
        "brutal boxing sports movie turbulent life outside ring almost tragedy character not likable pitiable brutal",
        "classic adventure war movie psychological drama british officer fight ottoman success ego dangerous"
    ]
}

df = pd.DataFrame(data)

# Calculate movie review vectors
movie_reviews = df.pivot_table(index='movie', columns='review', aggfunc=len, fill_value=0)
movie_vectors = movie_reviews.to_numpy()

# Calculate cosine similarity between movie vectors
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_sim = dot_product / (norm_a * norm_b)
    return cosine_sim

num_movies = movie_vectors.shape[0]
similarities = np.zeros((num_movies, num_movies))

for i in range(num_movies):
    for j in range(i + 1, num_movies):
        similarities[i, j] = cosine_similarity(movie_vectors[i], movie_vectors[j])
        similarities[j, i] = similarities[i, j]

# Find top 3 pairs of similar movies
num_top_pairs = 3
top_pairs = []

for _ in range(num_top_pairs):
    max_similarity = np.max(similarities)
    max_indices = np.unravel_index(np.argmax(similarities), similarities.shape)
    
    movie1 = df['movie'][max_indices[0]]
    movie2 = df['movie'][max_indices[1]]
    
    top_pairs.append((movie1, movie2, max_similarity))
    
    # Set the similarity to a very low value to find the next max
    similarities[max_indices] = -1.0

# Print the top similar movie pairs
for i, pair in enumerate(top_pairs):
    print(f"Top {i + 1} Similar Movies: {pair[0]} and {pair[1]} (Similarity: {pair[2]:.4f})")
