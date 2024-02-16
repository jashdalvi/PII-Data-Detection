pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)
mkdir data/
kaggle datasets download -d jashdalvi99/pii-dataset -p data/
kaggle datasets download -d nbroad/pii-dd-mistral-generated -p data/
python setup.py
rm data/pii-dataset.zip
rm data/pii-dd-mistral-generated.zip
python -m spacy download en_core_web_sm
git config --global user.name "jashdalvi"
git config --global user.email "jashdalvi99@gmail.com"
apt-get update && apt-get install tmux -y