from nltk.tokenize import word_tokenize as wt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from autocorrect import Speller
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

stemmer = PorterStemmer()
spell = Speller()
data = pd.read_csv(r'D:\Spam-Classifier-using-naive-bayes\spam.csv', encoding='ISO-8859-1')

data.head()
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

data.rename(
    columns={
        'v1': 'Label',
        'v2': 'Message'
    }, inplace=True)

# Visualize in Word Cloud the SPAM WORDS
spam_words = ' '.join(list(data[data['Label'] == 'spam']['Message']))
spam_wc = WordCloud(width=500, height=500).generate(spam_words)
plt.figure(figsize=(10, 8))
plt.imshow(spam_wc)
plt.show()

# Visualize in Word Cloud the NONE SPAM WORDS
ham_words = ' '.join(list(data[data['Label'] == 'ham']['Message']))
ham_wc = WordCloud(width=500, height=500).generate(ham_words)
plt.figure(figsize=(10, 8))
plt.imshow(ham_wc)
plt.show()

data.head()
new_data = []

for i in tqdm(range(data.shape[0])):
    lines = data.iloc[i, 1]
    lines = re.sub('[^A-Za-z]', ' ', lines)
    lines = lines.lower()
    tokenized_lines = wt(lines)
    lines_processed = []
    for j in tokenized_lines:
        if j not in stopwords.words('english'):
            lines_processed.append(spell(stemmer.stem(j)))
    final_lines = ' '.join(lines_processed)
    new_data.append(final_lines)

data['Label'] = data['Label'].apply(lambda x: 1 if x == 'spam' else 0)

Y = data['Label']

X_train, X_test, Y_train, Y_test = train_test_split(new_data, Y, test_size=0.25)

matrix = CountVectorizer()
X_train_vect = matrix.fit_transform(X_train).toarray()
X_test_vect = matrix.transform(X_test).toarray()
model = GaussianNB()
model.fit(X_train_vect, Y_train)

Y_pred = model.predict(X_test_vect)

print(accuracy_score(Y_test, Y_pred)*100)
