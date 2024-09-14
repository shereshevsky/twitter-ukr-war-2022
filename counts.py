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
from prefect import task, Flow
from collections import defaultdict, Counter
from prefect.executors import DaskExecutor
from typing import Tuple, Any, List


HOME = Path.home()
S3_BUCKET = os.environ.get("S3_BUCKET")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")


@task(name="Read User Data", checkpoint=False)
def read_twits(data_file: set) -> dict:

    logger = prefect.context.get("logger")
    logger.info(f"Start reding twits from {data_file}")
    total_tweets = 0
    res = defaultdict(lambda: defaultdict(int))
    s3_client = boto3.client("s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)

    s3file = s3_client.get_object(Key=data_file, Bucket=S3_BUCKET)
    file_data = s3file['Body'].read()
    with GzipFile(fileobj=io.BytesIO(file_data)) as f:
        try:
            for line in tqdm(f.readlines()):
                data = ujson.loads(line.decode().strip("\n"))
                for tweet in data["data"]:
                    res[tweet["author_id"]][tweet["lang"]] += 1
                    total_tweets += 1
        except Exception as e:
            logger.error(f"Error reading file {data_file}, error: {e}")
    logger.info(f"Total twits parsed - {total_tweets}")
    logger.info(f"Collected {len(res)} users")
    logger.info(f"Average user twits - {round(sum(len(i) for i in res.values())/len(res), 2)}")
    ujson.dump(res, (HOME / os.environ.get("RESULTS_PREFIX") / f"{data_file}-counts.pkl").open("wt"))
    return res


@task(name="Collect Files")
def collect_files(s3_path: str = "twits/") -> list:
    logger = prefect.context.get("logger")
    s3_client = boto3.client("s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_path)
    s3_files = response["Contents"]
    files = [i["Key"] for i in s3_files if i["Key"].endswith(".gz")]
    logger.info(f"Discovered files - {files}")
    return files


@task(name="Combine Counts")
def combine_counts(counts: List):
    combined = defaultdict(Counter)
    for count in tqdm(counts):
        for user in count:
            combined[user] += Counter(count[user])
    pickle.dump(combined, (HOME / os.environ.get("RESULTS_PREFIX") / "combined.pkl").open("wb"))


EXECUTOR = DaskExecutor(
    cluster_kwargs={"n_workers": 1, "threads_per_worker": 1}
)


def twitter_flow():
    with Flow("User Stats",
              # run_config=RUN_CONFIG
              executor=EXECUTOR
              ) as f:
        files = collect_files()
        counts = read_twits.map(files)
        combine_counts(counts)
    return f


if __name__ == '__main__':
    flow = twitter_flow()
    flow.run()
