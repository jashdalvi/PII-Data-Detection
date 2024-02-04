pip install -r requirements.txt
python setup.py
kaggle competitions download -c pii-detection-removal-from-educational-data -p data/
cd data
unzip pii-detection-removal-from-educational-data.zip
rm pii-detection-removal-from-educational-data.zip
cd ..
python -m spacy download en_core_web_sm
git config --global user.name "jashdalvi"
git config --global user.email "jashdalvi99@gmail.com"
apt-get update && apt-get install tmux -y