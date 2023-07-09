import json 
import numpy as np 
from sklearn.model_selection import train_test_split
import random 
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization,Dense, Dropout,concatenate, Input
from tensorflow.keras.optimizers import Adam

def load_data(json_file_path):
    with open(json_file_path,"r") as df:
        data = json.load(df)
        
    X_mfcc = np.array(data["mfccs"])
    X_zcr = np.array(data["zrc"])
    y = np.array(data["labels"])
    genres = data["genres"]

    print("MFCC shape is " + str(X_mfcc.shape))

    print("ZCR shape is " + str(X_zcr.shape))

    print("Y shape is " + str(y.shape))

    return X_mfcc,X_zcr, y, genres

def prepare_data(test_size, val_size):

    file_path = r"PATH_TO_JSON.json"

    X_mfcc,X_zcr,y, genres = load_data(file_path)

    X_mfcc_train, X_mfcc_test,X_train_zcr,X_zcr_test, y_train, y_test = train_test_split(X_mfcc,X_zcr,y,test_size = test_size)
    X_mfcc_train,X_mfcc_val,X_train_zcr,X_val_zcr, y_train, y_val = train_test_split(X_mfcc_train,X_train_zcr,y_train,test_size = val_size)

    # Add axis for channels to mfcc
    X_mfcc_train = X_mfcc_train[...,np.newaxis]
    X_mfcc_test = X_mfcc_test[...,np.newaxis]
    X_mfcc_val = X_mfcc_val[...,np.newaxis]

    # Add axis for channels to zcr
    X_train_zcr = X_train_zcr[...,np.newaxis]
    X_zcr_test = X_zcr_test[...,np.newaxis]
    X_val_zcr = X_val_zcr[...,np.newaxis]


    print("MFCC shape is " + str(X_mfcc_train.shape))

    print("ZCR shape is " + str(X_train_zcr.shape))

    print("Y shape is " + str(y_train.shape))


    return [X_mfcc_train, X_mfcc_test, X_mfcc_val],[X_train_zcr, X_zcr_test, X_val_zcr], [y_train, y_test, y_val], genres

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


def create_double_model(input_shape_spectrogram, input_shape_zcr, amnt_of_cats):

    # Spectrogram branch
    spectrogram_input = Input(shape=input_shape_spectrogram, name='spectrogram_input')
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(spectrogram_input)
    x = MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = 'same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = 'same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, kernel_size=(2, 2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides = (2,2), padding = 'same')(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Zero crossing rate branch
    zcr_input = Input(shape=input_shape_zcr, name='zcr_input')
    y = Dense(16, activation='relu')(zcr_input)  # You can modify this to better suit your data
    y = Flatten()(y)

    # Combine the output of the two branches
    combined = concatenate([x, y])

    # Output layer
    z = Dense(amnt_of_cats, activation='softmax')(combined)

    # Create model
    model = Model(inputs=[spectrogram_input, zcr_input], outputs=z)

    return model


def main():
    print("Loading")
    # Get all the traiing, test and validation sets
    mfccs,zcr,y, genres = prepare_data(0.2,0.1)

    # Select the amount of features you want to use to train
    # 1 = Only Mfcc
    # 2 = MFCC and Zero Crossing Rate

    amnt_of_features = 1


    # If you are going to use only MFCC

    if amnt_of_features ==  1:
        print("Creating Model")
        # Create the DL model
        input_shape = (mfccs[0].shape[1:])

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
        num_epochs = 20

        history = model.fit(mfccs[0],y[0], validation_data=(mfccs[2],y[2]), batch_size = 32,epochs=num_epochs)

    #Evaluate 
        error, acc = model.evaluate(mfccs[1],y[1])


        print("Accuracy is " + str(acc))

    # Do prediction for 20 random songs 

        for i in range(20):
            rand_ix = random.randint(0, len(mfccs[1]-1))
            rand_X = mfccs[1][rand_ix]
            rand_y = y[1][rand_ix]
            rand_y = int(rand_y)
            y_predicted = int(np.argmax(model.predict(rand_X[np.newaxis,...]),axis=1))

            print("The genre we think it is " + str(genres[y_predicted]) + ". The actual genre is " + str(genres[rand_y]))

    # If you are going to use both Zero Crossing Rate and MFCC
    elif amnt_of_features == 2:
        print("Creating Model")
        # Create the DL model
        input_mfcc_shape = (mfccs[0].shape[1:])
        input_zcr_shape = (zcr[0].shape[1:])
        print("Input MFCC Shape is: " + str(input_mfcc_shape))
        print("Input ZCR Shape is: " + str(input_zcr_shape))

        print(tf.config.list_physical_devices('GPU'))

        amnt_of_cats = len(genres)
        model = create_double_model(input_mfcc_shape, input_zcr_shape, amnt_of_cats)

        model.summary()

        # Compile the model 
        lr = 0.0002
        opt = Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


        # Train
        print("Training")
        num_epochs = 10
        history =model.fit([mfccs[0], zcr[0]], y[0], validation_data=([mfccs[2], zcr[2]], y[2]), epochs=num_epochs, batch_size=32)


    #Evaluate 
        error, acc = model.evaluate([mfccs[1],zcr[1]],y[1])


        print("Accuracy is " + str(acc))

    # Do prediction for 20 random songs 

        for i in range(20):
            rand_ix = random.randint(0, len(mfccs[1]-1))
            rand_mfcc = mfccs[1][rand_ix]
            rand_zcr = zcr[1][rand_ix]
            rand_y = y[1][rand_ix]
            rand_y = int(rand_y)
            y_predicted = int(np.argmax(model.predict([rand_mfcc[np.newaxis,...],rand_zcr[np.newaxis,...]]),axis=1))

            print("The genre we think it is " + str(genres[y_predicted]) + ". The actual genre is " + str(genres[rand_y]))



if __name__ == "__main__":
    main()