import re, pandas as pd, os
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')


def clean_text(text:str):
    """clean a single revew string.
    1: Lowercase
    2: remove html tags
    3: review special chars
    4: strip extra spaces 
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>"," ",text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ",text)
    text = text.strip()
    return text

def load_imdb():
    HF_TOKEN = os.getenv('HF_TOKEN')
    """
    Load IMDB dataset from HuggingFace and return
    train and test as pandas DF
    """
    dataset = load_dataset("stanfordnlp/imdb", token=HF_TOKEN)
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset['test'])
    return train_df, test_df


def preprocess(df: pd.DataFrame):
    """
    Apply cleaning to a DF and return it
    Keeps Original text in 'text_raw' for reference
    """
    df = df.copy()
    df['text_raw'] = df['text']
    df['text'] = df['text'].apply(clean_text)
    df['text_length'] = df['text'].apply(len)
    return df

def save_data(train_df: pd.DataFrame, test_df = pd.DataFrame):
    #Save preprocessed dataframes to data/ Folder as CSV
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index= False)
    print("saved train and test data to data/")

if __name__ == "__main__":
    print("Loading dataset ...")
    train_df, test_df = load_imdb()

    print("PreProcessing...")
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    print("sample of cleaned text")
    print(train_df['text'].iloc[0][:300])

    print("\nSaving")
    save_data(train_df, test_df)

    print("\nDone")
    print(f"train shape: {train_df.shape}")
    print(f"test shape: {test_df.shape}")


    