#Basic Image Classification - Trains a neural network model to classify images of clothing


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#Set fashipn_mnist to "file path"
fashion_mnist = tf.keras.datasets.fashion_mnist

#Loading data returns 4 numpy arrays(28 x 28 array holding values from 0 - 255 representing pixel shade)  train - teaches ML model to identify images  test - test to see if model is accurate
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Not included in data set so we need to define
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Array Specs
#(60000, 28, 28) - 60000 images with 28 x 28 pixels
print(train_images.shape)
#60000
print(len(train_labels))
#[9 0 0 ... 3 0 5] - 60000 assigned labels to each picture
print(train_labels)
#(10000, 28, 28) - test_images has 10000 images with 28 x 28 pixels
print(test_images.shape)
#10000 Each image in test is assigned a label
print(len(test_labels))

#opens up empty graph window
plt.figure()
#Selects the first image from traiing data and graphs it
plt.imshow(train_images[0])
#adds in color bar ranging from 0 to 255
plt.colorbar()
#removes grid lines
plt.grid(False)
plt.show()

print("Before scaling:")
print(test_images[1][1])

#Scaling down the pixels to be from a range of 0 - 1 instead of 0 - 255 (is this called feature scailing?(Talked about in ML class))
train_images = train_images / 255.0

test_images = test_images / 255.0

print("After scaling:")
print(test_images[1][1])

#Creates a new blank plot of size 10 by 10 inches
plt.figure(figsize=(10,10))

#prints out each of the first 25 pictures
for i in range(25):
    plt.subplot(5,5,i+1)
    #Takes away x and y axis hashes, leaves just the pic
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    #Plots the image at that index and makes it on a binary scale(black to white)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    #puts label underneath subplot for each image
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#Line 73 is creating an empty neural network with no layers (A LITTLE IFFY)
model = tf.keras.Sequential([
    #This is defining a layer that flattens our multidimensional data to 1 line of data(Im assuming it will just make the 28 x 28 array into a 1 D array)
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    #When I look this up it says that reLU's job is to set negative inputs to 0, but our range is from 0 to 1, how is this helpful if we have no negative values?
    tf.keras.layers.Dense(128, activation='relu'),
    #This returns a logit array, that represents raw unnormalized scores with length 10(0 being t-shirt, and 10 representing ankle boot)(I understand what it is doing but not what is actually happening in the background)
    tf.keras.layers.Dense(10)
])


#I am not familar of the different optimizers/loss functuions/metrics that are available, but I'm assuming we use whichever ones best fit for the problem we are handeling
#This is the compilation step of the neural network:
#We are slecting the optimizer algorithim we want to use, in this case ADAM(Adaptive Moment Estimation(No idea what this means))
model.compile(optimizer='adam',
              #A loss function is ultimately trying to determine how well the ML model is working, we want to minimize this loss function
              #from_logits = True is indicating that the data is raw logits and to run softmax internally(Not really sure what this means)
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #For this metric we are choosing to calculate the models accuracy
              metrics=['accuracy'])


#This begins the training for the machine learning model: we are passing all the training images and training labels into model to "fit" the model to this training data
#epochs=10, tells the model to run through this data 10 times, the more times we run through the dtaa, the better the accuracy of the model
model.fit(train_images, train_labels, epochs=10)  #91 %


#This is evaluating the performance of our trained machine learning model
#These files have not been seen by the model yet and is being tested based on the knowledge it had learn witht the previous training data.
#Verbose=2 is telling how much info to display during the evaluation process, in this example: 2(Loss, and accuracy)
#test_loss and test_acc will hold the models average loss and overall accuracy of the model after evaluating the test data
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)




#We are creating a new model to make predictions on images. Softmax will convert the models linear outputs(logits) to probabilities
#This addsa softmax layer to our existing model, normalizes logits to sum to 1. Giving it a probability
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

#Call new modle and let it try to predict labels for each image and sets all these outputs to 'predictions'
predictions = probability_model.predict(test_images)

#From my understanding each index of predictions holds an array of size 10, for each type article of clothing, with each index representing the probability of that image being that type of article of clothing
#In result, each index at predictions[0] will show the individual probabilities of that image being a certain article of clothing
#These probabilities for each image will sum to 1
print(predictions[0])

#This will print the index of the maximum probability(This corresponds to the type of clothing labels)
#This will print 9
print(np.argmax(predictions[0]))
#This will print the actual number it should be(which is 9 - Ankle Boot(Which is correct))
print(test_labels)





#Plots the image for a spedcific index in test_images
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
    #prints black and white image
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  #if they prediction is right make it blue, else red
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
    #Makes label under image the corresponding name to their prediction ,then the percentage to corresponding probability prediction, and then the correct corresponding name
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
  

#Plots graph which shows the probabilities of the specific image
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    #Sets xticks from 0-9
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    #Makes predicted label red and true label blue
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')



#Calls functions and prints the first picture
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()




# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# Add the image to a batch where it's the only member.
#Because keras only takes lists, or multiple inputs, in order to test with one image, we need to add it to a list by itself
#This example adds a dimension to the beginning of the array
img = (np.expand_dims(img,0))
#(28, 28)
print(img.shape)


#now call ooour model on this single image
predictions_single = probability_model.predict(img)
#(1, 28, 28)
print(predictions_single)


#Craete a plot with our prediction of image
plot_value_array(1, predictions_single[0], test_labels)
#make the x ticks allign with the corresponding names of clothing, rotate names 45 degrees, range is 0-9
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()


#Displays the max probability's index, which will show which index it should correspond to namesIin this case pullover)
print(np.argmax(predictions_single[0]))




#OVERALL UNDERSTANDING 