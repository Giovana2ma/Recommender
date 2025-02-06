# Movie Recommender System  

## Overview  
This project implements a content-based movie recommender using movie metadata from OMDB. The goal is to generate personalized recommendations by leveraging both user-item interactions and item content information.  

## Implementation  
The system uses FunkSVD, a matrix factorization approach to capture latent features, and the Rocchio algorithm, which refines user profiles based on relevance feedback. Different content representations and user profiling strategies were tested to improve recommendation quality.  

## Usage  
$ python3 -m venv rc2
$ source rc2/bin/activate
$ pip3 install -r /path/to/requirements.txt
$ python3 main.py ratings.jsonl content.jsonl targets.csv
