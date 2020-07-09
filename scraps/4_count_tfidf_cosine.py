from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Train Dataset --------------------------------------------------------------------------------------------------

dataset = ["the house had a tiny little mouse",
      "the cat saw the mouse",
      "the mouse ran away from the house",
      "the cat finally ate the mouse",
      "the end of the mouse story"]

print('Train Dataset : ')
print('---------------------------------------')
print(dataset)
print('')

# Count Vectorizer --------------------------------------------------------------------------------------------

cv_model = CountVectorizer(stop_words='english')
X = cv_model.fit_transform(dataset)

print('Count Vectorizer result')
print('---------------------------------------')
print(cv_model.vocabulary_)
# print(X)
print(X.toarray())
print('')

# TF-IDF Vectorizer ---------------------------------------------------------------------------------------------

tfidf_model = TfidfVectorizer(stop_words='english')
X = tfidf_model.fit_transform(dataset)

print('TF - IDF result')
print('---------------------------------------')
print(tfidf_model.vocabulary_)
# print(X)
print(X.toarray())
print('')

# Cosine Similarity -----------------------------------------------------------------------------------------------

q = 'mouse in the house'
dataset.append(q)

tfidf_model = TfidfVectorizer(stop_words='english')
X = tfidf_model.fit_transform(dataset)

print('The Cosine Similarity')
print('---------------------------------------')
print(cosine_similarity(X[-1], X))