from typing import Tuple, Any, List

from dotenv import load_dotenv
load_dotenv()

import os
import io
import boto3
import prefect
from tqdm import tqdm
from prefect import task, Flow, Parameter, unmapped
from prefect.engine.results import LocalResult
import spacy
import re
import ujson
import pandas as pd
from pathlib import Path
from gzip import GzipFile
from emoji import UNICODE_EMOJI
from collections import Counter, defaultdict
from gensim.models.phrases import Phrases, Phraser, FrozenPhrases

nlp = spacy.load('en_core_web_sm')

text_in_brackets = re.compile("\[(.*?)\]")
non_alpha = re.compile("[^A-Za-z']+")

HOME = Path.home()

S3_BUCKET = os.environ.get("S3_BUCKET")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")

results = LocalResult(dir=str(HOME / os.environ.get("RESULTS_PREFIX")))

s3_client = boto3.client("s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)

sentiment = pd.read_csv(HOME / 'Documents/osi/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', delimiter='\t',
                        header=None, names=['word', 'emotion', 'score'])

most_f_60k = set(pd.read_csv("./data/wordlist_60k_g803.txt", sep="\t").lemma.values)

positive = list(sentiment[(sentiment.emotion == 'positive') & (sentiment.score != 0)].word.values)
negative = list(sentiment[(sentiment.emotion == 'negative') & (sentiment.score != 0)].word.values)
anger = list(sentiment[(sentiment.emotion == 'anger') & (sentiment.score != 0)].word.values)
anticipation = list(sentiment[(sentiment.emotion == 'anticipation') & (sentiment.score != 0)].word.values)
disgust = list(sentiment[(sentiment.emotion == 'disgust') & (sentiment.score != 0)].word.values)
fear = list(sentiment[(sentiment.emotion == 'fear') & (sentiment.score != 0)].word.values)
joy = list(sentiment[(sentiment.emotion == 'joy') & (sentiment.score != 0)].word.values)
sadness = list(sentiment[(sentiment.emotion == 'sadness') & (sentiment.score != 0)].word.values)
surprise = list(sentiment[(sentiment.emotion == 'surprise') & (sentiment.score != 0)].word.values)
trust = list(sentiment[(sentiment.emotion == 'trust') & (sentiment.score != 0)].word.values)


@task(name="Read Text From Tweeter Files", checkpoint=True, result=results, target="{data_file}_texts.df")
def read_data(data_file: str) -> Tuple[list, list, list]:
    logger = prefect.context.get("logger")
    logger.info(f"Start reding twits from {data_file}")
    all_texts = []
    emoji_texts = []
    translated_emoji_texts = []

    s3file = s3_client.get_object(Key=data_file, Bucket=S3_BUCKET)
    file_data = s3file['Body'].read()
    with GzipFile(fileobj=io.BytesIO(file_data)) as f:
        try:
            for line in tqdm(f.readlines()):
                data = ujson.loads(line.decode().replace("\n", " "))
                for tweet in data.get("data", []):
                    lang = tweet.get("lang")
                    if lang != 'en':
                        continue
                    referenced_tweets = tweet.get("referenced_tweets", [])
                    if any([i.get("type") == "retweeted" for i in referenced_tweets]):
                        continue
                    raw_text = tweet.get("text")
                    cleaned_text = non_alpha.sub(' ', raw_text).lower()
                    if any([i in raw_text for i in UNICODE_EMOJI["en"]]):
                        emoji_texts.append(cleaned_text)
                        translated_emoji = " ".join(
                            [UNICODE_EMOJI["en"][i].lower().replace(":", "").replace("_", " ")
                             for i in UNICODE_EMOJI["en"] if i in raw_text])
                        translated_emoji_texts.append(translated_emoji + " " + cleaned_text)

                    all_texts.append(cleaned_text)
        except Exception as e:
            logger.error(f"Error reading file {data_file}, error: {e}")

    def cleaning(i, doc):
        _txt = [token.lemma_.lower().strip()
                if token.lemma_ != "-PRON-"
                else token.lower_
                for token in doc if not token.is_stop
                and len(token.lemma_) > 1
                and token.lemma_ != 'amp'
                ]  # tokenize, remove stopwords, lowercase
        if len(_txt) > 1:
            return ' '.join(_txt)

    all_texts = [cleaning(i, doc) for i, doc in enumerate(nlp.pipe(texts=all_texts, batch_size=6000, n_process=4))]
    emoji_texts = [cleaning(i, doc) for i, doc in enumerate(nlp.pipe(texts=emoji_texts, batch_size=6000, n_process=4))]
    translated_emoji_texts = [cleaning(i, doc) for i, doc in enumerate(nlp.pipe(texts=translated_emoji_texts, batch_size=6000, n_process=4))]

    logger.info(f"Finished reading twits from {data_file} - {len(all_texts), len(emoji_texts), len(translated_emoji_texts)}")

    return all_texts, emoji_texts, translated_emoji_texts


@task(name="Check Emotions", checkpoint=True, result=results, target="{data_file}_sentiment.df")
def check_emotions(data_file, texts):
    logger = prefect.context.get("logger")
    logger.info(f"Checking emotions for {data_file}")

    all_texts, emoji_texts, translated_emoji_texts = texts

    def check_emotions(corpus):
        all_tokens = [j for i in corpus if i for j in i.split(" ") if j]
        known_only = [i for i in all_tokens if i in most_f_60k]
        cnt = Counter(known_only)

        return {
            emotion:
                sum([cnt[i] for i in eval(emotion)])/len(known_only)
            for emotion in ["positive", "negative", "anger", "anticipation",
                            "disgust", "fear", "joy", "sadness",
                            "surprise", "trust"]
        }

    emotions_all = check_emotions(all_texts)
    emotions_emoji = check_emotions(emoji_texts)
    emotions_emoji_translated = check_emotions(translated_emoji_texts)

    return emotions_all, emotions_emoji, emotions_emoji_translated


@task(name="Collect Files")
def collect_files(s3_path: str) -> list:
    logger = prefect.context.get("logger")
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_path)
    s3_files = response["Contents"]
    files = [i["Key"] for i in s3_files]
    logger.info(f"Discovered files - {files}")
    return files


@task(name="calc_word_frequency_and_clean_texts",
      checkpoint=True, result=results,
      target="{data_file}_frequency_and_clean_texts.df")
def calc_word_frequency_and_clean_texts(data_file, texts: list) -> (Counter, list):
    clean_texts = [[j for j in i.split(" ") if j in most_f_60k and len(i) >= 2 and i != 'amp'] for i in texts[1] if i]
    # all_tokens = [j for i in texts[1] if i for j in i.split(" ") if j]
    # known_only = [i for i in all_tokens if i in most_f_60k]
    return Counter([j for i in clean_texts for j in i]), clean_texts


@task(name="combine word frequency")
def combine_word_freq(texts: list) -> dict:
    res = {k: v for d in texts for k, v in d.items()}
    df = pd.DataFrame(res)
    df2 = df.T
    df2.index = ["-".join([j.replace(".jsonl.gz", "") for j in i.split("-")[2:6]]) for i in df2.index.values]
    top_1k = df2.sum(axis=0).sort_values(ascending=False).index[:1000]

    return df2.loc[:, top_1k]


@task(name="calc_emotion_frequency", result=results, target="{file}_ngram_frequency.df")
def tokens_frequency_with_ngrams(file: str, word_freq_and_texts: dict, phraser: FrozenPhrases) -> Counter:
    phraser = Phrases.load(str(HOME / os.environ.get("RESULTS_PREFIX") / "trigram.phrases"))
    ngram_texts = []
    freq, texts = word_freq_and_texts[file]
    ngram_texts.extend([j for i in texts for j in phraser[i]])
    return Counter(ngram_texts)


@task(name="calc_ngrams", checkpoint=True, result=results, target="trigram.df")
def calc_ngrams(word_freq_and_texts: List):
    texts = []
    for file_freq_and_texts in word_freq_and_texts:
        word_freq, texts = list(file_freq_and_texts.values())[0]
        texts.extend(texts)
    bigram_phrases = Phrases(texts, min_count=30, progress_per=10000, max_vocab_size=200000)
    bigram = Phraser(bigram_phrases)
    trigram_phrases = Phrases(bigram[texts], min_count=30, progress_per=10000, max_vocab_size=200000)
    trigram = Phraser(trigram_phrases)
    trigram.save(str(HOME / os.environ.get("RESULTS_PREFIX") / "trigram.phrases"))
    return trigram


@task(name="calc_ngrams_and_clean_texts", checkpoint=True, result=results, target="global_ngram_frequency.df")
def calculate_global_frequency(files: list, ngram_freq: List[Counter]):
    k = 100
    merged = defaultdict(list)
    for file, freq in zip(files, ngram_freq):
        merged[file[-22:-9]] = freq
    df = pd.DataFrame(merged)
    terms_over_k = df.sum(axis=1)[df.sum(axis=1) > k].index
    df2 = df.loc[terms_over_k]
    return df2


def twitter_flow():
    with Flow("Twitter") as f:
        p_path = Parameter(name='p_path', required=True)
        files = collect_files(p_path)
        texts = read_data.map(files)
        # word_freq_and_texts = calc_word_frequency_and_clean_texts.map(files, texts)
        sentiments = check_emotions.map(files, texts)
        # trigram_model = calc_ngrams(word_freq_and_texts)
        # ngram_frequency = tokens_frequency_with_ngrams.map(files, word_freq_and_texts, unmapped(trigram_model))
        # global_ngram_freq = calculate_global_frequency(files, ngram_frequency)
    return f


if __name__ == '__main__':
    flow = twitter_flow()
    flow.run(parameters={"p_path": ""})

