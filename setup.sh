# setup.sh
#!/bin/bash
export PATH=/usr/bin:$PATH

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --quiet

mkdir output/

python main.py
