""" Create first network with Keras """
from keras.models import Sequential
from keras.layers import Dense
import numpy
import numpy as np
""" Keras to use """
from keras.optimizers import Adam

""" Hyperopt plus Keras """
from hyperas.distributions import uniform
""" dropout regularization """
from keras.layers import Dropout
""" scan part for hyperas """
from hyperas.distributions import choice, uniform, conditional
from hyperopt import space_eval
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.utils import eval_hyperopt_space
from hyperas import optim

""" to save model JSON and HDF5 """
from keras.models import model_from_json

""" Inform best results """
from keras.callbacks import ModelCheckpoint


""" fix random seed for reproducibility """
seed = 7
numpy.random.seed(seed)
""" load pima indians dataset """
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
""" split into input (X) and output (Y) variables """
X = dataset[:,0:8]
Y = dataset[:,8]
def data():
  dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
  X = dataset[:,0:8]
  Y = dataset[:,8]
  return X, Y, X, Y

""" create model """
def model(X ,Y):
  model = Sequential()
  model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
  model.add(Dense(8, init='uniform', activation='relu'))
  model.add(Dropout({{uniform(0, 1)}}))
  """ model.add(Dropout(0.2)) """
  model.add(Dense(8, init='uniform', activation='relu'))
  model.add(Dense(1, init='uniform', activation='sigmoid'))
  
  

  """  keras.optimizers.Adam try learning rate decay """
  adam_decay=Adam(lr={{uniform(0,1)}}, decay=1.3)#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  """ Compile model """
  model.compile(loss='binary_crossentropy', optimizer=adam_decay, metrics=['accuracy'])
  """ checkpoint """
  filepath="weights.best.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor= ' val_acc ' , verbose=1, save_best_only=True, mode= ' max ' )
  callbacks_list = [checkpoint]



  """ Fit the model """
  history=model.fit(X, Y, nb_epoch={{choice([4, 10,5])}}, batch_size=10,callbacks=callbacks_list,  verbose=2)

  """ list all data in history """
  print "history.history.keys()"
  print(history.history.keys())
  print "history.history.keys()[0]",history.history.keys()[0]
  print  "history.history[ ' ", history.history.keys()[0] ,"' ]", history.history[ history.history.keys()[0] ]
  print ""
  print  "history.history[ ' ", history.history.keys()[1] ,"' ]", history.history[ history.history.keys()[1] ]
  print ""

  """ serialize model to JSON """
  """ https://machinelearningmastery.com/save-load-keras-deep-learning-models/ """ 
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
  """ serialize weights to HDF5 """
  model.save_weights("model.h5")
  print("Saved model to disk")
  score, acc = model.evaluate(X,Y, verbose=0)
  print('Test accuracy:', acc)
  return {'loss': -acc, 'status': STATUS_OK, 'model': model}



if __name__ == '__main__':
    trials=Trials()
    best_run, best_model, space = optim.minimize(model=model, data=data,algo=tpe.suggest,max_evals=5,trials=trials, eval_space=True, return_space=True)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print "\n"
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print "\n"
    for t, trial in enumerate(trials):
        vals = trial.get('misc').get('vals')
        print("Trial %s vals: %s" % (t, vals))
        print "eval_hyperopt_space",eval_hyperopt_space(space, vals)
print "\n"
for trial in trials.trials:
  print "Trials"
  """ print "result",trial['result'],"misc",trial['misc'] """
  print trial
