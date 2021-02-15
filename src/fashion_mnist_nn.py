import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class MyCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if logs.get('acc')>0.99:
			print("Reached accuracy greater than 0.99!")
			self.model.stop_training = True

callback = MyCallback()

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images.reshape(training_images.shape[0],28,28,1)
training_images = training_images/255.0
test_images = test_images.reshape(test_images.shape[0],28,28,1)
test_images = test_images/255.0

model = tf.keras.Sequential([
	Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
	MaxPooling2D(2, 2),
	Conv2D(32, (3,3), activation='relu'),
	MaxPooling2D(2, 2),
	Flatten(),
	Dense(128, activation='relu'),
	Dense(10, activation='softmax')
	])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5, callbacks=[callback])
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
