# Tensorflow Feature Extraction

## Setup
Generate Virtual environment
```bash
python3 -m venv ./venv
```
Enter environment
```bash
source venv/bin/activate
```
Install required libraries
```bash
pip install -r requirements.txt
```
Extracting features using VGG
```
python3 features.py
```
Train model -d <path to dataset> -o <path to hdf5 file>
```bash
python3 train.py -d <path to hdf5 file>
```
