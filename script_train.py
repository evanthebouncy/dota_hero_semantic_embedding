from model import *
import sys

# start the training process some arguments
embednet = EmbedNet(tf.Session())
restart = True

def train_model(embednet, epoch=3000):

  if restart:
    embednet.initialize()
  else:
    embednet.initialize()
    embednet.load_model('./models/embednet.ckpt')

  for i in xrange(epoch):
    embednet.train(rand_datas(N_BATCH))

    if i % 20 == 0:
      embednet.save()

train_model(embednet, 10000)

