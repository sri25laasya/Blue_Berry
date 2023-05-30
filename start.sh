pip install -q kaggle
mkdir  ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle competitions download -c playground-series-s3e14
unzip playground-series-s3e14.zip

pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:BLUE_BERRY --reload #uvicorn is starting our file(web server)