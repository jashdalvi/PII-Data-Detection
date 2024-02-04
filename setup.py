import zipfile

zipFileName = "/root/PII-Data-Detection/data/pii-detection-removal-from-educational-data.zip"
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
