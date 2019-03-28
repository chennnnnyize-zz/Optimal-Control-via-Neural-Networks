import tensorflow as tf
import keras
import numpy as np
from numpy import shape
from keras.optimizers import SGD
from Neural_Net_Module_convex import rnn_model
import csv
import matplotlib.pyplot as plt

lr = 0.01
alpha = 0.5
No_epochs = 2
batch_size = 200
nb_epoch = 30
controllable_dim=16
seq_length = 1

################ Reshape data as RNN sequence #################
###############################################################
def reorganize(X_train, Y_train, seq_length):
    
    # X_train
    x_data = []
    for i in range(len(X_train) - seq_length):
        x_new = X_train[i:i + seq_length]
        x_data.append(x_new)
    
    # Y_train
    y_data = Y_train[seq_length:]
    y_data = y_data.reshape((-1, 1))

    return x_data, y_data


################ Main Function for prediction ####################
##################################################################
if __name__ == '__main__':
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print ("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'tf', temporarily setting to 'th'")

    sess = tf.Session()
    keras.backend.set_session(sess)

    #Read in EnergyPlus building simulation data
    with open('building_data.csv', 'r') as csvfile: 
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        rows = rows[1:]
    
    # Print dataset shape and feature dimension
    print("Dataset shape", shape(rows))
    rows = np.array(rows[1:], dtype=float)
    feature_dim = rows.shape[1]
    print("Feature dimension",feature_dim)

    # Normalize building data features and response
    max_value = np.max(rows,axis=0)
    print("Max power values: ", max_value)
    min_value=np.min(rows,axis=0)
    rows2=(rows-min_value)/(max_value-min_value) #row2: data after normalization in [0,1] interval

    # Reorganize the X and Y to the RNN sequence
    X_data, Y_data = reorganize(rows2[:, 0:feature_dim-1], rows2[:, feature_dim-1], seq_length = seq_length)
    print("Overall dataset shape", shape(X_data))
    print("Check if there is any Data is NAN:",np.argwhere(np.isnan(X_data)))
    X_data=np.array(X_data, dtype=float)
    Y_data=np.array(Y_data, dtype=float)
    
    # Train-Test data splitting 80%/20%
    train_len = int(43226*0.8)
    X_train = X_data[0:train_len]
    Y_train = Y_data[0:train_len]
    print('Number of training samples', Y_train.shape[0])
    X_test = X_data[train_len:-1]
    Y_test = Y_data[train_len:-1]
    print('Number of testing samples', Y_test.shape[0])
    
    # Define tensor in Tensorflow. x is a 3-D tensor, 
    # which corresponds to single-value energy consumption y
    x = tf.placeholder(tf.float32, shape=(None, seq_length, feature_dim-1))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    
    # Define the RNN model: RNN parameter defined in "utils_building.py"
    model = rnn_model(seq_length=seq_length, input_dim=feature_dim-1)
    predictions = model(x)
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True) #Stochastic Gradient Descent methods for model training
    model.compile(loss='mean_squared_error',optimizer=sgd)

    # Fit the RNN model with training data and save the model weight
    # X_train: training input
    # Y_train: training output
    # batch_size: batch size for stochastic gradient descent to accelerate training
    # epochs: iteration times
    model.fit(X_train, Y_train, batch_size=batch_size,epochs=30, shuffle=True) # validation_split = 0.1
    model.save_weights('rnn_clean.h5')
    print("Clean training completed!")
    
    # Use the fitted model on test data set
    Y_pred = model.predict(X_test, batch_size=200)
    
    # Plot the prediction result for 2 weeks
    t = np.arange(0,2016)
    plt.plot(t,Y_test[3000:5016],'r--',label="True")
    plt.plot(t,Y_pred[3000:5016],'b',label="predicted")
    plt.legend(loc='best')
    plt.xlabel("Day")
    plt.ylabel("Electricity Consumption (normalized)")
    ax = plt.gca() # grab the current axis
    ax.set_xticks(144*np.arange(0,14)) # choose which x locations to have ticks
    ax.set_xticklabels(["1","2","3","4","5","6","7","8","9","10","11","12","13","14"]) # set the labels to display at those ticks
    plt.title("Building electricity consumption on test dataset")
    plt.show()
    
    # Calculate the RMSE
    RMSE = np.sqrt(np.mean(((Y_pred-Y_test)**2)))
    print("Test RMSE:", RMSE)
