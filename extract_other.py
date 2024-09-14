from dotenv import load_dotenv

load_dotenv()

import os
import io
import boto3
import ujson
import prefect
import pickle
from tqdm import tqdm
from pathlib import Path
from gzip import GzipFile
from prefect import task, Flow, unmapped
from collections import defaultdict, Counter
from prefect.executors import DaskExecutor
from typing import Tuple, Any, List


HOME = Path.home()
S3_BUCKET = os.environ.get("S3_BUCKET")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")


@task(name="Read User Data", checkpoint=False)
def read_twits(data_file: set, users: tuple):
    logger = prefect.context.get("logger")

    texts = []

    logger.info(f"Start reding twits from {data_file}")

    s3_client = boto3.client("s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)

    s3file = s3_client.get_object(Key=data_file, Bucket=S3_BUCKET)
    file_data = s3file['Body'].read()

    with GzipFile(fileobj=io.BytesIO(file_data)) as f:
        try:
            for line in tqdm(f.readlines()):
                data = ujson.loads(line.decode().strip("\n"))
                for tweet in data["data"]:
                    if tweet["author_id"] not in users:
                        texts.append(tweet["text"])

        except Exception as e:
            logger.error(f"Error reading file {data_file}, error: {e}")

    pickle.dump(texts, (HOME / os.environ.get("RESULTS_PREFIX") / f"{data_file}-other.pkl").open("wb"))


@task(name="Collect Files")
def collect_files(s3_path: str = "twits/") -> list:
    logger = prefect.context.get("logger")
    s3_client = boto3.client("s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_path)
    s3_files = response["Contents"]
    files = [i["Key"] for i in s3_files if i["Key"].endswith(".gz")]
    logger.info(f"Discovered files - {files}")
    return files


@task(name="Get Users")
def get_uk_ru_users(data_path: str = "/Users/anodot_1/Documents/osi"):
    return pickle.load(open(f"{data_path}/users_ru.pkl", "rb")), pickle.load(open(f"{data_path}/users_uk.pkl", "rb"))


EXECUTOR = DaskExecutor(
    cluster_kwargs={"n_workers": 4, "threads_per_worker": 1}
)


def twitter_flow():
    with Flow("User Stats",
              # run_config=RUN_CONFIG
              executor=EXECUTOR
              ) as f:
        files = collect_files()
        users = get_uk_ru_users()
        read_twits.map(files, unmapped(users))
    return f


if __name__ == '__main__':
    flow = twitter_flow()
    flow.run()
