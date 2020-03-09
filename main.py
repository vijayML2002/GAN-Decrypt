%tensorflow_version 2.x

import hashlib
import tensorflow as tf
import time
from IPython import display

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

#def the lib
import itertools
import numpy as np

#defining the hashing function
word_dict = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5,
             'f':6, 'g':7, 'h':8, 'i':9, 'j':10,
             'k':11, 'l':12, 'm':13, 'n':14, 'o':15,
             'p':16, 'q':17, 'r':18, 's':19, 't':20,
             'u':21, 'v':22, 'w':23, 'x':24, 'y':25,
             'z':26}

def convert(x):
    letters = []

    for l in x:
        lett = word_dict[l]
        lett = str(lett)
        if len(lett)==1:
            lett = '0' + lett
        letters.append(lett)

    concat = ''

    for l in letters:
        concat = concat + l

    return concat


all_possible = []
all_possible_hash = []

test_list = [chr(x) for x in range(ord('a'), ord('z') + 1)]
d ={'1':test_list, '2':test_list, '3':test_list, '4':test_list}
for combo in itertools.product(*[d[k] for k in sorted(d.keys())]):
    all_possible.append(''.join(combo))

#number of all_possible word
print('The number of possible word -- {}'.format(len(all_possible)))

#creating the hash
for l in all_possible:
    all_possible_hash.append(convert(str(l)))

assert len(all_possible)==len(all_possible_hash)
print('The number of hash -- {}'.format(len(all_possible_hash)))

#printing out some value
for i in range(20):
    rand = np.random.randint(0,len(all_possible))
    print('{}  ---  {}  ---  {}'.format(all_possible[rand],all_possible_hash[rand],len(all_possible_hash[rand])))


all_possible = np.array(all_possible).reshape(len(all_possible),1)
all_possible_hash = np.array(all_possible_hash).reshape(len(all_possible_hash),1)

print('Shape of the all_possible      -  {}'.format(all_possible.shape))
print('Shape of the all_possible_hash -  {}'.format(all_possible_hash.shape))


train_x = []

for word in all_possible:
  a = ''
  for letter in word[0]:
    a = a + letter + ' '
   
  train_x.append(a)

train_x = np.array(train_x)
print('The shape of the train_x -- {}'.format(train_x.shape))


train_y = []

for num in all_possible_hash:
  a = ''
  for digit in num[0]:
    a = a + digit + ' '

  train_y.append(a)

train_y = np.array(train_y)
print('The shape of the train_y -- {}'.format(train_y.shape))

def tokenize(x):
  tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False)
  tokenizer.fit_on_texts(x)
  return tokenizer.texts_to_sequences(x),tokenizer

sample_pro , sample_tok = tokenize('hi') 
print(sample_pro)

def preprocess(x,y):
  process_x , tok_x = tokenize(x)
  process_y , tok_y = tokenize(y)

  return process_x,process_y,tok_x,tok_y

process_x,process_y,tok_x,tok_y = preprocess(train_x,train_y)

#cvt to arr
process_x = np.array(process_x)
process_y = np.array(process_y)


#train
process_train_x = process_x[:440000]
process_train_y = process_y[:440000]

#test
process_test_x = process_x[440001:450000]
process_test_y = process_y[440001:450000]

dataset_train = tf.data.Dataset.from_tensor_slices((process_train_x,process_train_y))
dataset_test  = tf.data.Dataset.from_tensor_slices((process_test_x,process_test_y))


#shuffling

dataset_train = dataset_train.shuffle(30000)
dataset_test = dataset_test.shuffle(300)


#defining the thief network

def theif():

  model_theif = tf.keras.models.Sequential()
  model_theif.add(tf.keras.layers.Input(shape = process_y.shape[1:]))
  model_theif.add(tf.keras.layers.Dense(128,activation='relu'))
  model_theif.add(tf.keras.layers.Dropout(0.5))
  model_theif.add(tf.keras.layers.Dense(64,activation='relu'))
  model_theif.add(tf.keras.layers.Dense(4))

  return model_theif


#defining the police network

def police():

  inp = tf.keras.layers.Input(shape=process_y.shape[1:])
  tar = tf.keras.layers.Input(shape=process_x.shape[1:])
  x = tf.keras.layers.concatenate([inp,tar])
  #model_police.add(tf.keras.layers.Input(shape=process_y.shape[1:]))
  x = tf.keras.layers.Dense(64,activation='relu')(x)
  last = tf.keras.layers.Dense(1)(x)

  return tf.keras.Model(inputs=[inp,tar],outputs=last)


#shape of the model

theif_net = theif()
tf.keras.utils.plot_model(theif_net,show_shapes=True,dpi=64)


out , inp = next(iter(dataset_train))

inp = tf.reshape(inp,[1,8])
inp = tf.cast(inp,dtype='float32')

out = tf.reshape(out,[1,4])
out = tf.cast(out,dtype='float32')


theif_out = theif_net(tf.reshape(inp,[1,8])) #checking police_net

print('Prediction')
print(theif_out)


#shape of the model

police_net = police()
tf.keras.utils.plot_model(police_net,show_shapes=True,dpi=64)


pol_out = police_net([out,inp],training=False)
print('Prediction')
print(pol_out)

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def police_loss(real,fake):
  real_loss = loss_obj(tf.ones_like(real),real)
  fake_loss = loss_obj(tf.zeros_like(fake),fake)
  total_loss = real_loss + fake_loss
  return total_loss


LAMBDA =  100

def theif_loss(fake,gen_output,target):
  fake_loss = loss_obj(tf.ones_like(fake),fake)
  l1_loss = tf.reduce_mean(tf.abs(target-gen_output))
  total_gen_loss = fake_loss + (LAMBDA * l1_loss)
  return total_gen_loss,fake_loss,l1_loss

theif_opt = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)
police_opt = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)


no_epoch = 50
dims = 4
no_example = 16
batch_size = 10

seed = tf.random.normal([no_example,dims])

def generate_data(model, test_input, tar):
  test_input = tf.reshape(test_input,[1,8])
  test_input = tf.cast(test_input,dtype='float32')
  prediction = model(test_input, training=True)

  display_list = [test_input[0], tar, prediction[0]]

  print('INPUT - {}  TAR - {}  PRED - {}'.format(display_list[0],display_list[1],display_list[2]))



for example_input, example_target in dataset_test.take(1):
  generate_data(theif_net, example_target, example_input)


@tf.function
def train_step(input_data,target,epoch):
  with tf.GradientTape() as gen_tape,tf.GradientTape() as dis_tape:
    input_data = tf.reshape(input_data,[1,8])
    input_data = tf.cast(input_data,dtype='float32')

    target = tf.reshape(target,[1,4])
    target = tf.cast(target,dtype='float32')

    theif_output = theif_net(input_data, training=True)

    pol_real_output = police_net([input_data, target], training=True)
    pol_generated_output = police_net([input_data, theif_output], training=True)
    
    gen_total_loss, gen_gan_loss, gen_l1_loss = theif_loss(pol_generated_output, theif_output, target)
    disc_loss = police_loss(pol_real_output, pol_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            theif_net.trainable_variables)
    discriminator_gradients = dis_tape.gradient(disc_loss,
                                                police_net.trainable_variables)
    
    theif_opt.apply_gradients(zip(generator_gradients,
                                            theif_net.trainable_variables))
    police_opt.apply_gradients(zip(discriminator_gradients,
                                                police_net.trainable_variables))

    with summary_writer.as_default():
      tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
      tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
      tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
      tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_data(theif_net, example_target, example_input)
    print("Epoch: ", epoch)

    # Train
    for n, (input_data, target) in train_ds.enumerate():
      train_step(target, input_data, epoch)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))

fit(dataset_train,no_epoch,dataset_test) #fitting the model
