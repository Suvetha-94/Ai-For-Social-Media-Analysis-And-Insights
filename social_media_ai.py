import re
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud

# Collect multiple tweets/reviews from the user
print("Enter your tweets/reviews one by one. Type 'done' when finished:")

tweets = []
while True:
    tweet = input("> ")
    if tweet.strip().lower() == 'done':
        break
    if tweet.strip():  # avoid empty input
        tweets.append(tweet.strip())

if not tweets:
    print("No input provided. Exiting.")
    exit()

# Create DataFrame with the user input tweets
data = {
    'username': [f'@user{i+1}' for i in range(len(tweets))],
    'tweet': tweets
}
df = pd.DataFrame(data)

# Function to clean tweet text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)     # remove mentions
    text = re.sub(r'#', '', text)        # remove hashtag symbol
    text = re.sub(r'[^A-Za-z\s]', '', text)  # remove special characters
    text = text.lower().strip()
    return text

# Apply cleaning
df['clean_tweet'] = df['tweet'].apply(clean_text)

# Function to get sentiment label
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Apply sentiment analysis
df['sentiment'] = df['clean_tweet'].apply(get_sentiment)

# Print analyzed data
print("\nAnalyzed Tweets and Sentiments:")
print(df[['tweet', 'sentiment']])

# Plot sentiment distribution
sentiment_counts = df['sentiment'].value_counts()

plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Generate word cloud
all_words = ' '.join(df['clean_tweet'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Tweets')
plt.show()
