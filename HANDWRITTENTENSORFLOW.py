'''
importing tensorflow
create the mnist_database
split the data into x,y-train, x,y-test
create the neural network model, with flatten layer(take image 28X28 pixels and flatten it to (784,1)input array)
add dense layer with 128 neurons in the hidden layer, with activation "relu"
at the end add Dense layer with 10 outputs(0-9)
create predicitions(without any training yet)
implement the softmax algorithm
create the loss (cost) function and optimize it with adam compiler
fit (train) the model
then evaluate and predict again
'''
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(128, activation = 'relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
predictions = model(x_train[:1]).numpy()
print(predictions)
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
