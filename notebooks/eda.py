import pandas as pd, numpy as np, os
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()


HF_TOKEN = os.getenv('HF_TOKEN')
dataset = load_dataset("stanfordnlp/imdb", token = HF_TOKEN)
print(dataset)

train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"training data shape {train_df.shape}")
print(f"testing data shape {test_df.shape}")

#looking at heads

print(f"training head: {train_df.head(5)}")
print(f"testing head {test_df.head(5)}")

#check for nulls

print(train_df.isnull().sum())

#check class distribution

print(train_df['label'].value_counts(normalize=True))

#checking review length and stats
train_df['text_length'] = train_df['text'].apply(len)
print(train_df['text_length'].describe())


#sample  review
print("Sample Positive review")
print(train_df[train_df['label'] == 1]['text'].iloc[0][:300])

print("Sample Negative review")
print([train_df[train_df['label'] == 0]['text'].iloc[0][:300]])


