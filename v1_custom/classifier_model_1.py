import numpy as np
import tensorflow as tf

# Generate example data
num_samples = 100
input_size = 100
output_size = 3

# Create random input data and labels, scaled to [0, 1]
input_data = np.random.rand(num_samples, input_size)
labels = np.random.randint(output_size, size=num_samples)

# Normalize input data to [0, 1] range
input_data = (input_data - input_data.min()) / \
    (input_data.max() - input_data.min())

# Define the custom classification function


def custom_classification(output_probs):
    # Apply your custom function here
    # For example, you might choose the index of the maximum probability as the classification
    return np.argmax(output_probs)

# Define the classifier neural network model


class ClassifierModel(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super(ClassifierModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(
            64, activation='relu', input_shape=(input_size,))
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(
            output_size, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


# Initialize the classifier model
classifier = ClassifierModel(input_size, output_size)

# Compile the model
classifier.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with supervised learning
num_epochs = 20
batch_size = 16
classifier.fit(input_data, labels, epochs=num_epochs, batch_size=batch_size)

# Test the trained model
test_input = np.random.rand(1, input_size)
# Normalize test input to [0, 1] range
test_input = (test_input - test_input.min()) / \
    (test_input.max() - test_input.min())
predicted_probs = classifier.predict(test_input)
custom_class = custom_classification(predicted_probs[0])

print(f"Input: {test_input}")
print(f"Predicted Probabilities: {predicted_probs}")
print(f"Custom Classification: {custom_class}")
