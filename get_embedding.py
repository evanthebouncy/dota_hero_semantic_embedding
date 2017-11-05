from model import *
from data import N_BATCH
import pickle

embednet = EmbedNet(tf.Session())
embednet.load_model('./models/embednet.ckpt')

def get_hero(idx):
  ret = []
  for i in range(N_BATCH):
    input_array =  [0.0 for _ in range(L)]
    input_array[idx] = 1.0
    ret.append(input_array)
  return np.array(ret)

hero_embeddings = dict()

for i in range(1, 121):
  hero = get_hero(i)
  hero_vec = list(embednet.get_embedding(hero)[0][0])
  hero_embeddings[i] = hero_vec

print hero_embeddings

pickle.dump(hero_embeddings, open("hero_embeddings.p", "wb"))
