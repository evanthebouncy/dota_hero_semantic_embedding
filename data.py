import pickle
import random
import numpy as np

L = 121
N_BATCH = 50

teams = pickle.load(open( "teams.p", "rb" ) )
print len(teams)
dataN = len(teams)

def rand_draft():
  team = random.choice(teams)
  return [int(x) for x in random.choice(team[:2]).split(',')]

def rand_io():
  draft = rand_draft()
  rand_idx = random.randint(0,4)
  rand_hero = draft[rand_idx]
  draft.remove(rand_hero)
  return rand_hero, draft

def rand_data():
  input_array =  [0.0 for _ in range(L)]
  output_array = [[0.0, 1.0] for _ in range(L)]
  rand_hero, rand_team = rand_io()
  input_array[rand_hero] = 1.0
  for r_hero in rand_team:
    output_array[r_hero] = [1.0, 0.0]
  return input_array, output_array

def rand_datas(NN):
  inputz, outputz = [], []
  for _ in range(NN):
    ii, oo = rand_data()
    inputz.append(ii)
    outputz.append(oo)
  return np.array(inputz), np.array(outputz)

if __name__ == "__main__":
  print rand_datas(2)
  
