pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)
mkdir data/
kaggle datasets download -d jashdalvi99/pii-dataset -p data/
kaggle datasets download -d nbroad/pii-dd-mistral-generated -p data/
kaggle datasets download -d tonyarobertson/mixtral-original-prompt -p data/
kaggle datasets download -d mpware/pii-mixtral8x7b-generated-essays -p data/
kaggle kernels output conjuring92/pii-mv01-aux -p data/
# kaggle datasets download -d jashdalvi99/pii-data-detection-deberta-v3-large-cv-098156 -p data/
python setup.py
rm data/pii-dataset.zip
rm data/pii-dd-mistral-generated.zip
rm data/mixtral-original-prompt.zip
rm data/pii-mixtral8x7b-generated-essays.zip
# rm data/pii-data-detection-deberta-v3-large-cv-098156.zip
python -m spacy download en_core_web_sm
git config --global user.name "jashdalvi"
git config --global user.email "jashdalvi99@gmail.com"
apt-get update && apt-get install tmux -y