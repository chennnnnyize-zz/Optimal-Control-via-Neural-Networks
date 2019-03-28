##############################
#The Optimization code for building HVAC system
#Dependencies: Neural_Net_Module.py, training dataset, trained model (optional)
#Hyper-parameters: 
#  lr: the learning rate
#  nb_epoch: The number of initial training epochs
#  controllable_dim: the dimension of vectors partial optimization operates on.
#  batch_size: training batch size
#  TEMP_MAX, TEMP_MIN: the constraints on temperature setpoints
#  seq_length: the inertia considered for building system
#Code design:
#(a) Initial training/load trained model: to set up the temporal machine learning model for building HVAC dynamics
#(b) Optimization w.r.t inputs: Do gradient descents on input controllable variables
#(c) Optimization algorithm performance analysis
#############################

import tensorflow as tf
import keras
import numpy as np
from numpy import shape
from keras.optimizers import SGD
from Neural_Net_Module import rnn_model
import csv
import matplotlib.pyplot as plt

lr = 0.01
batch_size = 200
nb_epoch = 30
controllable_dim=16
seq_length= 36
TEMP_MAX = 24
TEMP_MIN = 20

def scaled_gradient(x, predictions, target, tset, tref_high, tref_low):
    ##################################################################
    #Function to calculate the gradient and do gradient descents w.r.t inputs controllable variables
    # x -- input
    # prediction -- predicted electricity consumption
    # target -- target electricity consumption
    # tset -- controllable part in x
    # tref_high -- upper bound for controllables
    # tref_low -- lower bound for controllables
    ##################################################################
    
    #Loss function: sum((prediction-target)^2)
    mis_tracking_target=tf.square(predictions- target)
    
    #Take gradient with respect to x_{T}, since it contains all the x value needs to be updated
    grad, = tf.gradients(mis_tracking_target, x)
    
    #Define the gradient of log barrier function on constraints
    #loss_comfort=tf.square(tf.sub(tset, tref))
    grad_comfort_high = 1/((tref_high - tset))
    grad_comfort_low = 1/((tset - tref_low))
    grad_contrained = grad[:,:,0:16] + 0.000000001*(grad_comfort_high+grad_comfort_low)   
    return grad, grad_contrained


def reorganize(X_train, Y_train, seq_length):
    # Organize the input and output to feed into RNN model
    x_data = []
    for i in range(len(X_train) - seq_length):
        x_new = X_train[i:i + seq_length]
        x_data.append(x_new)
    
    # Y_train
    y_data = Y_train[seq_length:]
    y_data = y_data.reshape((-1, 1))

    return x_data, y_data

def check_control_constraint(X, dim, uppper_bound, lower_bound):
    for i in range(0, shape(X)[0]):
        for j in range(0, shape(X)[0]):
            for k in range(0, dim):
                if X[i,j,k] >= uppper_bound[i,j,k]:
                    X[i,j,k] = uppper_bound[i,j,k] - 0.01
                if X[i,j,k] <= lower_bound[i,j,k]:
                    X[i,j,k] = lower_bound[i,j,k] + 0.01
    return X

if __name__ == '__main__':
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print ("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'tf', temporarily setting to 'th'")

    sess = tf.Session()
    keras.backend.set_session(sess)

    with open('building_data.csv', 'r') as csvfile: #good dataset/data2.csv
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        rows = rows[1:]
    print("Dataset shape", shape(rows))
    rows = np.array(rows[1:], dtype=float)

    feature_dim = rows.shape[1]
    print("Feature dimension",feature_dim)

    # Normalize the feature and response
    max_value = np.max(rows,axis=0)
    print("Max power values: ", max_value)
    min_value=np.min(rows,axis=0)
    rows2=(rows-min_value)/(max_value-min_value)

    #Reorganize to the RNN-like sequence
    X_train, Y_train = reorganize(rows2[:, 0:feature_dim-1], rows2[:, feature_dim-1], seq_length=seq_length)
    print("Training data shape", shape(X_train))
    print("X_train None:",np.argwhere(np.isnan(X_train)))
    X_train=np.array(X_train, dtype=float)
    Y_train=np.array(Y_train, dtype=float)

    # Test data: change here for real testing data
    Y_test = Y_train
    X_test = X_train
    print('Number of testing samples', Y_test.shape[0])
    print('Number of training samples', Y_train.shape[0])

    # Define tensor
    x = tf.placeholder(tf.float32, shape=(None, seq_length, feature_dim-1))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    tset = tf.placeholder(tf.float32, shape=(None, seq_length, controllable_dim))
    tref_high = tf.placeholder(tf.float32, shape=(None, seq_length, controllable_dim))
    tref_low = tf.placeholder(tf.float32, shape=(None, seq_length, controllable_dim))
    target = tf.placeholder(tf.float32, shape=(None, 1))

    # Define the tempture setpoint upper and lower bound
    temp_low = TEMP_MIN*np.ones((1,controllable_dim)) #temp setpoint lowest as 20
    temp_low = (temp_low-min_value[0:controllable_dim])/(max_value[0:controllable_dim]-min_value[0:controllable_dim])
    temp_high = TEMP_MAX*np.ones((1,controllable_dim))#temp setpoint highest as 25
    temp_high = (temp_high-min_value[0:controllable_dim])/(max_value[0:controllable_dim]-min_value[0:controllable_dim])
    
    # Define the RNN model, establish the graph and SGD solver
    model = rnn_model(seq_length=seq_length, input_dim=feature_dim-1)
    predictions = model(x)
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',optimizer=sgd)

    # Fit the RNN model with training data and save the model weight
    model.fit(X_train, Y_train, batch_size=batch_size,epochs=nb_epoch, shuffle=True) # validation_split=0.1
    model.save_weights('rnn_clean.h5')
    y_value = model.predict(X_test[0:5000], batch_size=32)

    # Record the prediction result
    with open('predicted_rnn2.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(y_value)

    with open('truth_rnn2.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(Y_test[0:5000])
    
    # Plot the prediction result. This is the same as Building_Load_Forecasting.py
    t = np.arange(0,2016)
    plt.plot(t,Y_test[216:216+2016],'r--',label="True")
    plt.plot(t,y_value[216:216+2016],'b',label="predicted")
    plt.legend(loc='northeast')
    ax = plt.gca() # grab the current axis
    ax.set_xticks(144*np.arange(0,14)) # choose which x locations to have ticks
    ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun","Mon","Tue","Wed","Thu","Fri","Sat","Sun"]) # set the labels to display at those ticks
    plt.title("Building electricity consumption")
    plt.show()
    print("Clean training completed!")
    print("Training percentage error:",np.mean(np.divide(abs(y_value-Y_train[0:5000]),Y_train[0:5000])))

    # Optimization step starts here!
    model.load_weights('rnn_clean.h5')
    y_value = model.predict(X_train, batch_size=32)

    X_new = []
    grad_new = []
    mpc_scope = seq_length
    X_train2 = X_train
    with sess.as_default():
        counter = 216
        for q in range(288):
            if counter % 100 == 0 and counter > 0:
                print("Time" + str(counter))

            # Define the control output target 
            Y_target = (0 * Y_test[counter:counter + mpc_scope]).reshape(-1, 1)
            
            # upper and lower bound for controllable features
            X_upper_bound = np.tile(temp_high,(mpc_scope, seq_length,1))
            X_lower_bound = np.tile(temp_low,(mpc_scope, seq_length,1))
            
            # Define input: x_t, x_{t+1},...,x_{t+mpc_scope}
            X_input = X_train2[counter:counter + mpc_scope]
            X_input = check_control_constraint(X_input, controllable_dim, X_upper_bound, X_lower_bound)
            X_controllable = X_input[:, :, 0:controllable_dim]
            
            # the uncontrollable part needs to be replaced by prediction later!!!
            X_uncontrollable = X_input[:, :, controllable_dim:feature_dim - 1] 
            
            # Initialize the SGD optimizer
            grad, grad_contrained = scaled_gradient(x, predictions, target, tset, tref_high, tref_low)

            X_new_group = X_input
            #Change 3 here: the iteration step
            for it in range(20):
                gradient_value, gradient_constrain = sess.run([grad, grad_contrained],feed_dict={x: X_new_group,
                                                            target: Y_target,
                                                            tset: X_controllable,
                                                            tref_high: X_upper_bound,
                                                            tref_low: X_lower_bound,
                                                            keras.backend.learning_phase(): 0})
                
                X_new_group[:, :, 0:controllable_dim] = X_new_group[:, :, 0:controllable_dim] - gradient_constrain   
                
                # check if the controllable variable exceeds the boundary
                X_new_group = check_control_constraint(X_new_group, controllable_dim, X_upper_bound, X_lower_bound)
                y_new_group = model.predict(X_new_group)

            if X_new == []:
                X_new = X_new_group[0].reshape([1,seq_length,feature_dim-1])
                grad_new = gradient_constrain[0]
            else:
                X_new = np.concatenate((X_new, X_new_group[0].reshape([1,seq_length,feature_dim-1])), axis=0)
                grad_new = np.concatenate((grad_new, gradient_constrain[0]), axis=0)
                print("Constrained gradient value:",gradient_constrain[0,-1,0:10],'Counter',counter-216)
            
            #Update the x value in the training data
            X_train2[counter] = X_new_group[0].reshape([1,seq_length,feature_dim-1])
            for i in range(1, seq_length):
                X_train2[counter+i,0:seq_length-i,:] = X_train2[counter,i:seq_length,:]
            
            #Next time step
            counter += 1

    X_new = np.array(X_new, dtype=float)
    y_new = model.predict(X_new, batch_size=64)

    t = np.arange(0,len(X_new))
    plt.plot(t,Y_test[216:216+len(X_new)],'r',label="True")
    plt.plot(t,y_value[216:216+len(X_new)],'b--',label="RNN prediction")
    plt.plot(t,y_new[0:len(X_new)],'g-.',label="Optimization")
    #plt.plot(t,np.mean(Y_train)*np.ones((len(X_new),1)),'y:',label="target")
    plt.legend(loc='best')
    plt.show()

    #Plot the result
    dime = 7
    t = np.arange(0,len(X_new))
    X_temp = rows[0:len(X_new),dime]
    X_temp_new = X_new[0:len(X_new),0,dime]*(max_value[dime]-min_value[dime])+min_value[dime]
    plt.plot(t,X_temp,'r--',label="previous")
    plt.plot(t,X_temp_new,'b',label="controlled")
    plt.legend(loc='best')
    ax = plt.gca() # grab the current axis
    ax.set_xticks(144*np.arange(0,8)) # choose which x locations to have ticks
    ax.set_xticklabels(["1","2","3","4","5","6","7","8"]) # set the labels to display at those tick
    plt.xlabel('Day')
    plt.ylabel('Temperature setpoint')
    plt.show()
    
    with open('y_new3.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(y_new)

    with open('x_new.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(X_new)

    with open('gradient.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(grad_new)

    print("Finished!")

