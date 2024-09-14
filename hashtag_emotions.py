from typing import List, Dict

from dotenv import load_dotenv

load_dotenv()

import os
import io
import boto3
import prefect
from tqdm import tqdm
from prefect import task, Flow, Parameter
from prefect.engine.results import LocalResult, S3Result
from prefect.storage import S3
from prefect.run_configs import LocalRun
from prefect.executors import LocalExecutor
from prefect.executors import DaskExecutor

import spacy
import re
import ujson
import pandas as pd
from pathlib import Path
from gzip import GzipFile
from collections import Counter, defaultdict
# from gensim.models.phrases import Phrases, Phraser, FrozenPhrases
from emoji import UNICODE_EMOJI


text_in_brackets = re.compile("\[(.*?)\]")
non_alpha = re.compile("[^A-Za-z']+")
hashtag_mappings = {    "belarus":    {"belarus",        "minsk",        "lukashenko",        "belarusian",        "belarusians",        "bielorussia"    },
                        "biden":    {"biden",        "joebiden",        "bidenisafailure",        "bidenisadisgrace",        "potus",        "bidenswar"    },
                        "crypto":    {"bitcoin",        "nft",        "crypto",        "btc",        "nfts",        "cryptocurrency",        "eth",        "nftcommunity",        "blockchain",        "nftart",        "dogecoin",        "nftcollection",        "nftgiveaway",        "nftdrop",        "nftartist",        "cryptonews",        "nftcollector",        "cryptocurrencies"    },
                        "cyber":    {"anonymous",        "osint",        "cybersecurity",        "cyber",        "cyberattacks",        "cyberattack",        "cyberwar"    },
                        "energy":    {"nordstream2",        "oil",        "gas",        "gazprom",        "energy"    },
                        "eu":    {"eu",        "europe",        "ue"    },
                        "israel":    {"israel",        "israeli"    },
                        "nato":    {"nato",        "otan"    },
                        "palestine":    {"palestine",        "freepalestine", "palestinian" },
                        "putin":    {"putin",        "stopputin",        "poutine",        "istandwithputin",        "putinwarcriminal",        "stopputinnow",        "putinisawarcriminal",        "putinswar",        "putinhitler",        "defeatputin",        "vladimirputin",        "kremlin",        "fckputin",        "fuckputin",        "putinwarcrimes",        "putinsgop",        "putler",        "fckptn",        "putinasesino",        "путин",        "stopputinswar",        "putinatwar",        "putinskrieg",        "blockputinwallets",        "putins",        "poetin",        "resistputin",        "criminalputin",        "heyputin",        "killputin",        "adolfputin", "vladimirpoutine" },
                        "sanctions":    {"swift",        "sanctions",        "sanctionrussianow",        "banrussiafromswift",        "sanktionen",        "swiftban", "boycottrussia", "blockputinwallets", "visa"},
                        "trump":    {"trump",        "traitortrump",        "putinspuppet",        "trumpisarussianasset",        "maga",        "donaldtrump",        "trumptraitor",        "putinspuppets",        "trumprussia",        "traitortrumpputinspuppet",        "trumpcult",        "trumpisatraitor"    },
                        "un":    {"un",        "unga",        "unitednations",        "оон"    },
                        "usa":    {"whitehouse", "usa",        "sotu",        "us",        "america",        "gop",        "sotu2022",        "stateoftheunion",        "unitedstates",        "american" , "washington"  },
                        "ww3":    {"worldwar3",        "wwiii",        "terceraguerramundial",        "nuclearwar",        "worldwariii",        "ww3"    },
                        "zelensky":    {"zelensky",        "zelenskyy",        "zelenskiy",        "volodymyrzelensky",        "selenskyj",        "volodymyrzelenskyy",        "zelenski",        "presidentzelensky",        "zelinsky"    },
                        "russia": {"rus", "russia", "kremiln", "russian", "stoprussia", "rusia", "russie", "russland", "stoprussianaggression", "russianarmy", "moscow","istandwithrussia","russians","oprussia", "stoprussianagression", "russiagohome", "rosja", "russes", "russe", "russiancolonialism", "moskova", "moscou", "rússia", "russiansoldiers"},
                        "ukraine": {"ukarine", "ukraine", "ukraina", "kyiv", "standwithukraine", "kiev", "ukrainian", "ukraineunderattack", "ukrainewar", "kharkiv", "ukraina", "ucraina", "istandwithukraine", "standwithukriane", "ukriane", "ukrayna", "prayforukraine", "standwithukraine️", "україна", "ukrania", "ukrainians", "украина", "westandwithukraine", "standingwithukraine", "istandwithukriane", "saveukraine", "freeukraine", "helpukraine", "ukraineunderattaсk", "ukrainewillresist", "kiew", "staywithukraine", "ukrain", "prayingforukraine", "oekraine", "kiyv", "ukrainestrong", "ukrainianarmy", "ukranie", "kyev"},
                        "war_peace": {"війна", "wearenato", "peace", "prayforpeace", "peacenotwar", "peaceforukraine", "peaceinukraine", "worldpeace", "russianukrainianwar", "ukrainerussianwar","stopwar", "ukrainerussiacrisis", "russiaukrainewar", "ukrainewar", "russiaukraineconflict", "ukrainerussiawar","russiaukrainecrisis", "ukrainecrisis", "ukrainerussie", "nowar", "russiainvadedukraine", "ukrainerussiaconflict", "russianinvasion", "ukraineconflict", "stopwarinukraine", "stopthewar", "russiaucraina", "wojna", "ucraniarussia", "russiainvadesukraine", "nowarinukraine", "nowarplease", "guerreenukraine", "russianaggression", "notowar", "stopputinswar", "putinatwar", "война", "saynotowar", "stopwars", "russiainvasion", "conflict", "antiwar"},
                        }

HOME = Path.home()
nlp = spacy.load('en_core_web_sm')

results = LocalResult(dir=str(HOME / os.environ.get("RESULTS_PREFIX")))
# s3_result = S3Result(
#     bucket=os.environ.get("S3_BUCKET"),
#     boto3_kwargs={
#         "aws_access_key_id": os.environ.get("ACCESS_KEY"),
#         "aws_secret_access_key": os.environ.get("SECRET_ACCESS_KEY")}
# )


@task(name="Read Text From Tweeter Files", checkpoint=True, result=results,
      target="prefect_results/{data_file}_hashtag_emotions_texts_wo_emojis.df")
def read_data(data_file: str) -> Dict[str, List[str]]:
    logger = prefect.context.get("logger")
    logger.info(f"Start reding twits from {data_file}")

    # sentiment = pd.read_csv(
    #     s3_client.get_object(Key="data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt",
    #                          Bucket=os.environ.get("S3_BUCKET"))['Body'],
    #     delimiter='\t', header=None, names=['word', 'emotion', 'score'])

    # most_f_60k = set(pd.read_csv(
    #     s3_client.get_object(Key="data/wordlist_60k_g803.txt", Bucket=os.environ.get("S3_BUCKET"))['Body'],
    #     sep="\t").lemma.values)
    #
    # positive = list(sentiment[(sentiment.emotion == 'positive') & (sentiment.score != 0)].word.values)
    # negative = list(sentiment[(sentiment.emotion == 'negative') & (sentiment.score != 0)].word.values)
    # anger = list(sentiment[(sentiment.emotion == 'anger') & (sentiment.score != 0)].word.values)
    # anticipation = list(sentiment[(sentiment.emotion == 'anticipation') & (sentiment.score != 0)].word.values)
    # disgust = list(sentiment[(sentiment.emotion == 'disgust') & (sentiment.score != 0)].word.values)
    # fear = list(sentiment[(sentiment.emotion == 'fear') & (sentiment.score != 0)].word.values)
    # joy = list(sentiment[(sentiment.emotion == 'joy') & (sentiment.score != 0)].word.values)
    # sadness = list(sentiment[(sentiment.emotion == 'sadness') & (sentiment.score != 0)].word.values)
    # surprise = list(sentiment[(sentiment.emotion == 'surprise') & (sentiment.score != 0)].word.values)
    # trust = list(sentiment[(sentiment.emotion == 'trust') & (sentiment.score != 0)].word.values)
    s3_client = boto3.client("s3", aws_access_key_id=os.environ.get("ACCESS_KEY"),
                             aws_secret_access_key=os.environ.get("SECRET_ACCESS_KEY"))


    texts = {k: [] for k in hashtag_mappings.keys()}

    s3file = s3_client.get_object(Key=data_file, Bucket=os.environ.get("S3_BUCKET"))
    file_data = s3file['Body'].read()
    with GzipFile(fileobj=io.BytesIO(file_data)) as f:
        try:
            for line in tqdm(f.readlines()):
                data = ujson.loads(line.decode().replace("\n", " "))
                for tweet in data.get("data", []):
                    lang = tweet.get("lang")
                    if lang != 'en':
                        continue

                    tags = [i.get("tag").lower() for i in tweet.get("entities", {"hashtags": []}).get("hashtags", [])
                            ] + [i.get("normalized_text").lower()
                                 for i in tweet.get("entities", {"annotations": []}).get("annotations", [])]

                    topics = [topic for topic in hashtag_mappings if any(tag in hashtag_mappings[topic] for tag in tags)]

                    if topics:
                        raw_text = tweet.get("text")
                        cleaned_text = non_alpha.sub(' ', raw_text).lower()
                        # if any([i in raw_text for i in UNICODE_EMOJI["en"]]):
                        #     cleaned_text += " " + " ".join(
                        #         [UNICODE_EMOJI["en"][i].lower().replace(":", "").replace("_", " ").replace('.', '')
                        #          for i in UNICODE_EMOJI["en"] if i in raw_text])
                        #
                        for topic in topics:
                            texts[topic].append(cleaned_text)

        except Exception as e:
            logger.error(f"Error reading file {data_file}, error: {e}")

    logger.info(f"Finished reading twits from {data_file} - {sum(len(i) for i in texts.values())}")

    return texts


@task(name="Clean texts", checkpoint=True, result=results,
      target="prefect_results/{data_file}_hashtag_emotions_texts_clean_wo_emojis.df")
def clean_text(data_file, text) -> Dict[str, List[str]]:
    def cleaning(doc):
        _txt = [token.lemma_.lower().strip()
                if token.lemma_ != "-PRON-"
                else token.lower_
                for token in doc if not token.is_stop
                and len(token.lemma_) > 1
                and token.lemma_ != 'amp'
                ]  # tokenize, remove stopwords, lowercase
        if len(_txt) > 1:
            return ' '.join(_txt)
    _results = {k: None for k in text.keys()}
    for topic in text:
        _results[topic] = [cleaning(doc) for doc in nlp.pipe(texts=text[topic], batch_size=6000)]

    return _results


@task(name="Collect Files")
def collect_files(s3_path: str = "twits/") -> list:
    logger = prefect.context.get("logger")
    s3_client = boto3.client("s3", aws_access_key_id=os.environ.get("ACCESS_KEY"),
                             aws_secret_access_key=os.environ.get("SECRET_ACCESS_KEY"))

    response = s3_client.list_objects_v2(Bucket=os.environ.get("S3_BUCKET"), Prefix=s3_path)
    s3_files = response["Contents"]
    files = [i["Key"] for i in s3_files]
    logger.info(f"Discovered files - {files}")
    return files


def twitter_flow():
    # with Flow(FLOW_NAME, storage=STORAGE, run_config=RUN_CONFIG, executor=EXECUTOR) as f:
    with Flow(FLOW_NAME, run_config=RUN_CONFIG, executor=EXECUTOR) as f:
        files = collect_files()
        texts = read_data.map(files)
        clean_text.map(files, texts)
        # word_freq_and_texts = calc_word_frequency_and_clean_texts.map(files, texts)
        # sentiments = check_emotions.map(files, texts)
        # trigram_model = calc_ngrams(word_freq_and_texts)
        # ngram_frequency = tokens_frequency_with_ngrams.map(files, word_freq_and_texts, unmapped(trigram_model))
        # global_ngram_freq = calculate_global_frequency(files, ngram_frequency)
    return f


FLOW_NAME = "hashtag_emotions"
# STORAGE = S3(
#     bucket=os.environ.get("S3_BUCKET"),
#     key=f"flows/{FLOW_NAME}.py",
#     stored_as_script=True,
#     # this will ensure to upload the Flow script to S3 during registration
#     local_script_path=f"./{FLOW_NAME}.py",
# )

EXECUTOR = DaskExecutor(
    cluster_kwargs={"n_workers": 4, "threads_per_worker": 1}
)

RUN_CONFIG = LocalRun(
    labels=["twitter"],
)

if __name__ == '__main__':
    flow = twitter_flow()
    flow.run()
    # flow.register(project_name="uk_war_twits")


