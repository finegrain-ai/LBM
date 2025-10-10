## Download the RORD dataset locally

```
pip install "huggingface_hub[cli]"
sudo apt-get install p7zip-full
bash download.sh
```

## Format the data to webdataset

ETA on AWS `c6i.8xlarge`: 2h30
```
python preprocessing.py
```

Result is saved in `data/RORD-processed/` (auto-created if not existing)