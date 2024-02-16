import zipfile

zipFileName = "/root/PII-Data-Detection/data/pii-dataset.zip"
zipPath = "/root/PII-Data-Detection/data/"
zf = zipfile.ZipFile(zipFileName)
zf.extractall(path=zipPath)
zf.close()

zipFileName = "/root/PII-Data-Detection/data/pii-dd-mistral-generated.zip"
zipPath = "/root/PII-Data-Detection/data/"
zf = zipfile.ZipFile(zipFileName)
zf.extractall(path=zipPath)
zf.close()

import nltk
import os
from dotenv import load_dotenv
load_dotenv()
nltk.download("stopwords")
nltk.download('punkt')
