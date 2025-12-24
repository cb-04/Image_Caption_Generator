import time
from tensorflow.keras.utils import get_file

def download_with_retry(url,filename,max_retries=3):
    for attempt in range(max_retries):
        try:
            return get_file(filename,url)
        except Exception as e:
            if attempt == max_retries-1:
                raise e
            print("Download attempt failed.")
            time.sleep(3)