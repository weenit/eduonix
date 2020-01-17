import tensorflow

# laod dataset
(x_train,y_train), (x_test,y_test) = tensorflow.keras.datasets.fashion_mnist.load_data()

# normalize data
x_train = x_train/255.0
x_test = x_test/255.0

# reshape data for model
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

# create model
model = tensorflow.keras.Sequential()
model.add(tensorflow.keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu', input_shape = (28,28,1)))
model.add(tensorflow.keras.layers.Conv2D(64, kernel_size = 3, activation='relu'))
model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.Dense(120, activation='relu'))
model.add(tensorflow.keras.layers.Dense(10, activation='softmax'))

# compile model
model.compile(optimizer = 'adam', loss="sparse_categorical_crossentropy", metrics = ['accuracy'])

# fit model
model_log = model.fit(x_train, y_train, batch_size=128, validation_data=(x_test, y_test), epochs=10)

# evaluate model
print(model.evaluate(x_test,y_test))
