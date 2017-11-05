# A semantic embedding of Dota 2 Heros

embed dota2 heros w2v style based on drafts

![alt text](https://raw.githubusercontent.com/evanthebouncy/dota_hero_semantic_embedding/master/embedded_pix/embed1.png)


## Word 2 Vec (what is this!?)

Word2Vec is an embedding technique in NLP (natural language proccessing) where,
using a large textual corpus, words are "embedded" into a dense subspace where
"similar" words are closer together. 

https://en.wikipedia.org/wiki/Word2vec

On a high level, given a large body of text (a corpus), two words are
considered "similar" if they have similar context of surrounding words. So for
example, if I have two sentences "I love my cat, he is cute" and "I love my
dog, he is cute", then "cat" and "dog" are similar to each other because they
share the same surrounding word context "I love \_\_\_, he is cute".

As you can see this process can be kind of whacky, because if I had another
sentence in the corpus saying "I love my boyfriend / girlfriend, he/she is
cute" then all of a sudden your boyfriend/girlfriend is similar to that of a
cat/dog. 

However, you would hope that there will be other sentences like "a cat is a
small four legged creature" and "a dog is a small furry companion", so that
overall, on average, "cat" is much similar to "dog" than your "boyfriend /
girlfriend".

## Word 2 Vec with respect to dota

With what word2vec in mind, let us see how this might be a useful notion in
Dota. How might we deem two heros similar to each other? Well, one way to do it
is if they share the same "role" in a team. For example, lion and rhasta are
very similar, both often drafted for being supports with good disables. That
being said, one would expect given a draft, one can substitute out lion for
rhasta, and vice versa.

Consider a draft [lion, ogre, OD, mirana, void], now we can substitute out lion
for rhasta, obtaining [rhasta, ogre, OD, mirana, void]. You see these two
drafts are not substantially different, and for that reason, lion and rhasta
are not substantially different.

Formally, the shared contexts is [ogre, OD, mirana, void] for both rhasta and
lion in the preceeding example. Because their contexts are similar, those two
heros are similar. Over thousands of games, we hope that this primitive notion
of similarity would manifest itself into strong patterns, such that we'll find
many interesting similarities.

## Training Procedure

For the technically inclined, this is the training procedure. Similar to how
W2V is trained, we will use a hero to predict its team. Concretely speaking,
given "lion", can we guess [ogre, OD, mirana, void]?  This almost impossible
task is exactly the training objective for our neural network. We feed it with
"lion", and we demand it to produce the other 4 heros on the team, [ogre, OD,
mirana, void]. 

Specifically, we first "embed" the input lion into a low-dimensional vector
space, say 20 dimension, and use this hidden vector to generate the full team
in turn. At first this embedding might seem like an extra step, but I will now
explain why it is actually very useful!

Remember that both "lion" and "rhasta" are tasked into generating the same
context of [ogre, OD, mirana, void]. By transforming them into a
low-dimensional vector space, we're losing information. That is to say, in the
embedded space, we no longer can remember "lion" and "rhasta" distinctly from
each other. The only way out of this is remembering something more general as
"support with strong disables" in the embedded space, this way the neural
network encodes similar information, but with less bits.  If the embedded
vector space sufficiently small, "lion" and "rhasta" will have no choice but to
be collapsed into similar low dimensional vector. This low dimensional vector,
the "embedding", is then used to generate similar team contexts.

For the really nitty gritty details, the input "lion" is encoded as a 1-hot
vector, where the size of the vector is 121, the total number of dota hero
indexes (some are missing but no matter). The output team [ogre, OD, mirana,
void] is encoded as a multi-hot bag-of-word vector, also of size 121, with 1
occuring at the index of ogre, OD, mirana, and void. The embedded space is
simply a 20 dimensional vector.

So there you have it, 1 hot input of dimension 121, 20 dimensional latent
representation, bag-of-word output of dimension 121. Error is measured as
xentropy on each output bit. Simple enough. 

The resulted embedding can be used for various fun stuff like similarity, or
plotted in 2D if we first reduce the dimension with tsne.

Incidentally, these are the top 4 similar hero to rhasta:

    (0.0, u'Shadow Shaman')
    (0.32796648, u'Keeper of the Light')
    (0.46663746, u'Rubick')
    (0.47017339, u'Dark Willow')
    (0.48853499, u'Lion')
    (0.50325406, u'Jakiro')

As you can see, Lion is not exactly the same, but it seems to be a group of supports.

We can also look at what are the similar hero to axe:

    (0.0, u'Axe')
    (0.3325673, u'Lycan')
    (0.36000824, u'Centaur Warrunner')
    (0.39020795, u'Lich')
    (0.39779869, u'Beastmaster')
    (0.40312102, u'Venomancer')
    (0.40369415, u'Riki')
    (0.4046663, u'Tusk')
    (0.41234338, u'Slardar')
    (0.41594386, u'Undying')

So perhapse a smattering of offlane heros and jungle heros.

You can modify the file query\_embeddings.py for these kind of similarities, you do not need any tensorflow as the embeddings are stored already in hero\_embeddings.p

## Some files and what they do if you want to mess with the code

### scrape_games.py
scrape opendota api for radiant/dire draft and winning team. For this project, only the drafts
are used, the winning / losing are ignored

### scrape_hero.py
scrape the opendota api for hero indexes

### data.py
sample a random team draft and produce the following input-output pairs

1-hot encoded single hero to a bag-of-word (multi-hot) encoded teammate drafts

### model.py
the neural network model as described, see code for more detail, I assume if you're looking at this you know what you're doing. The tricky part is the index-by-index xentropy loss function where some tensorflow hackery happens, the rest is simple enough.

### script_train.py
trains the model, stores the ckpt files under the models/ directory, which
doesn't exist in the repo cuz I didn't want to add the actual models to github,
it's fairly large

### get_embedding.py
save the embedding into a pickle file

### query_embedding.py
Play with this file for different embedding tasks such as finding which heros
are similar to another, also give the 2d t-sne representatino for
visualization. This is the most useful file to look at first.

### hero_embeddings.p
The stored pickle file for trained hero embeddings. You can use it as is, does not require tensorflow as it is already trained and stored here.

### drawings/
Some simple drawing on a html5 canvas. Change the file data\_vis.js to draw different pictures. There are some junk files in this directory I'm too lazy to clean up.

## Related Work
Prior approach such as 

http://federicov.github.io/word-embeddings-and-dota2.html

used Genism(sp?) to obtain an embedding. In our approach we encoded the full
training procedure with tensorflow end-2-end, and visualized the embedding
space with tsne. We also found out that a latent dimension of 50 is too large
as the network learned to memorize the individual heros rather than
generalizing across similar hero roles.

## some system details
Using python 2.7 and tensorflow 1.0.1 so something might not be compatible. This thing trains rather quickly, like 5 minutes or even less on my machine with some Nvidia960 GPU laptop thing.

Enjoy!!

--evanthebouncy
