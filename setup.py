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

zipFileName = "/root/PII-Data-Detection/data/mixtral-original-prompt.zip"
zipPath = "/root/PII-Data-Detection/data/"
zf = zipfile.ZipFile(zipFileName)
zf.extractall(path=zipPath)
zf.close()

zipFileName = "/root/PII-Data-Detection/data/pii-mixtral8x7b-generated-essays.zip"
zipPath = "/root/PII-Data-Detection/data/"
zf = zipfile.ZipFile(zipFileName)
zf.extractall(path=zipPath)
zf.close()

# zipFileName = "/root/PII-Data-Detection/data/pii-data-detection-deberta-v3-large-cv-098156.zip"
# zipPath = "/root/PII-Data-Detection/data/"
# zf = zipfile.ZipFile(zipFileName)
# zf.extractall(path=zipPath)
# zf.close()

import nltk
import os
from dotenv import load_dotenv
load_dotenv()
nltk.download("stopwords")
nltk.download('punkt')
