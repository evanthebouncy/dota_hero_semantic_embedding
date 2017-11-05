import pickle
import random
import numpy as np
from sklearn.manifold import TSNE

hero_embeddings = pickle.load(open("hero_embeddings.p", "rb"))
hero_idxs, idx_heros = pickle.load(open("hero_idxs.p", "rb"))

def get_dist(hero_id1, hero_id2):
  hero1_vec = np.array(hero_embeddings[hero_id1])
  hero2_vec = np.array(hero_embeddings[hero_id2])
  diff = hero1_vec - hero2_vec
  return np.dot(diff, diff)

def get_similar(hero_id):
  to_sort = [(get_dist(hero_id,other_idx), idx_heros[other_idx]) for other_idx in idx_heros]
  return list(sorted(to_sort))

def t_sne():
  hero_indexs = [idx for idx in idx_heros]
  X = []
  for idx in hero_indexs:
    X.append(hero_embeddings[idx])
  X = np.array(X)
  X_embedded = TSNE(n_components=2).fit_transform(X)
  hero_names = [idx_heros[idx] for idx in hero_indexs]
  return zip(hero_names, X_embedded)

if __name__ == '__main__':
  for x in get_similar(hero_idxs["Axe"]):
    print x

  tt = t_sne()
  to_print = dict()
  for name, coord in tt:
    name = str(name.replace(" ", "_").lower())
    to_print[name] = list(coord)

  print to_print
    
