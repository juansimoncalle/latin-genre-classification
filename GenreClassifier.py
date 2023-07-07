import json 
import numpy as np 
from sklearn.model_selection import train_test_split
import random 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed,Conv2D, MaxPooling2D, Flatten, BatchNormalization,Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_data(json_file_path):
    with open(json_file_path,"r") as df:
        data = json.load(df)
        
    X = np.array(data["mfccs"])
    y = np.array(data["labels"])
    genres = data["genres"]

    return X, y, genres

def prepare_data(test_size, val_size):

    file_path = r"PATH_TO_JSON.json"
    X,y, genres = load_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size)
    X_train,X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = val_size)
    X_train = X_train[...,np.newaxis]
    X_test = X_test[...,np.newaxis]
    X_val = X_val[...,np.newaxis]

    return X_train, X_test, X_val, y_train, y_test, y_val, genres

def create_model(input_shape, amnt_of_cats):
    model = Sequential()
    # Layer 1
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = 'same'))
    model.add(BatchNormalization())

    # Layer 2
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = 'same'))
    model.add(BatchNormalization())


    # Layer 3
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2), padding = 'same'))
    model.add(BatchNormalization())



    # Layer 4 
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    # output layer - 10 different categories
    model.add(Dense(amnt_of_cats, activation = 'softmax'))

    return model





def main():
    print("Loading")
    # Get all the traiing, test and validation sets
    X_train, X_test, X_val, y_train, y_test, y_val, genres = prepare_data(0.1,0.1)

    print("Creating Model")
    # Create the DL model
    input_shape = (X_train.shape[1:])

    print("Input Shape is: " + str(input_shape))

    print(tf.config.list_physical_devices('GPU'))

    amnt_of_cats = len(genres)
    model = create_model(input_shape,amnt_of_cats)
    model.summary()

    # Compile the model 
    lr = 0.005
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    # Train
    print("Training")
    num_epochs = 500
    history = model.fit(X_train,y_train, validation_data=(X_val,y_val), batch_size = 32,epochs=num_epochs)

    #Evaluate 
    error, acc = model.evaluate(X_test,y_test)


    print("Accuracy is " + str(acc))

    # Do prediction for 20 random songs 

    for i in range(20):
        rand_ix = random.randint(0, len(X_test-1))
        rand_X = X_test[rand_ix]
        rand_y = y_test[rand_ix]
        rand_y = int(rand_y)
        y_predicted = int(np.argmax(model.predict(rand_X[np.newaxis,...]),axis=1))

        print("The genre we think it is " + str(genres[y_predicted]) + ". The actual genre is " + str(genres[rand_y]))


if __name__ == "__main__":
    main()