import requests
import json
import time

hero_idxs = dict()
idx_heros = dict()

url = 'https://api.opendota.com/api/heroes'
r = requests.get(url)
for hero in r.json():
  hero_idx = hero["id"] 
  hero_name = hero["localized_name"]
  hero_idxs[hero_name] = hero_idx
  idx_heros[hero_idx] = hero_name

import pickle
pickle.dump((hero_idxs, idx_heros), open( "hero_idxs.p", "wb" ) )
