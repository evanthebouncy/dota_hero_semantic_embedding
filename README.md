# dota_hero_semantic_embedding
embed dota2 heros w2v style based on drafts

![alt text](https://raw.githubusercontent.com/evanthebouncy/dota_hero_semantic_embedding/master/embedded_pix/embed1.png)


# What is this

Word2Vec is an embedding technique in NLP where, using a large textual corpus, words are "embedded" into a dense subspace where "similar" words are closer together. 

On a high level, two words are considered "similar" if they have similar context of surrounding words. So for example, if I have two sentences "I love my cat, he is cute" and "I love my dog, he is cute", then "cat" and "dog" are similar to each other because they share the same surrounding word context "I love . . ., he is cute".


# scrape_games.py
scrape opendota api for radiant/dire draft and winning team, only the drafts are used, the winning / losing are ignored

# scrape_hero.py
scrape the opendota api for hero indexes

# data.py
sample a random team draft and produce the following input-output pairs

1-hot encoded single hero to a bag-of-word (multi-hot) encoded teammate drafts




Using python 2.7 and tensorflow 1.0.1 so something might not be compatible.
