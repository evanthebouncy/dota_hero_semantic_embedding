import requests
import json
import time

teams = []
seen = set()

for i in range(1000):
  try:
    time.sleep(0.5)
    print i, len(seen)
    url = 'https://api.opendota.com/api/publicMatches'
    if len(seen) > 0:
      url += '?less_than_match_id={}'.format(min(seen))
    r = requests.get(url)
    # print url
    for game in r.json():
      if game['match_id'] in seen:
        continue
      else:
        seen.add(game['match_id'])
        if game['game_mode'] == 22:
          teams.append( (game['radiant_team'], game['dire_team'], game['radiant_win']) )
  except:
    pass

import pickle
pickle.dump(teams, open( "teams.p", "wb" ) )
