# DSDA385_HW1
Machine Learning Capstone Homework 1

Dataset     Architecture        Accuracy        F1          Time to Train (s)                      Notes
Adult           MLP              0.7900        0.4142       6.296967506408691                      Using class weights or a custom loss  
                                                                                                   function could help the accuracy
Adult           CNN              0.7947        0.3112       15.32059931755066

CIFAR-100       MLP              0.2703        0.2632        147.31748056411743
CIFAR-10        CNN              0.4522        0.4370        492.013863325119                       A deeper model would do much better, 
                                                                                                   but I lack the computing time to do so

PCam            MLP             0.4958         0.2877       3016.0187394019273
PCam            CNN             0.5356          0.3287     3487.9128496092157


The objective of this homework assignment was to get comfortable in the pytorch library, to learn the differences in certain types of machine learning models, and to learn how these different models perform on different types of data. 
Each of my MLPs and CNNs were split into different files since I wanted them to be easily accessible and readable.

All layers with activation functions had ReLU functions as it is widely regarded as the best general activation function. The loss I used was binary cross entropy or cross entropy depending on if it was binary classification or not as that is what I am used to working with. All of my models will include an inverse pyramid style of neuron sizes, ex. 128, 64, 32, 16. This allows for more depth while not overfitting the model. All rows that include dropout, basically all that are not input and output layers, have a dropout rate of 0.2 to prevent some overfitting, but still keep most of the neurons available.

The adult data preprocessing was super interesting since it was a split of numerical and categorical data. Since basic machine learning models, like mine, cannot take categorical data, I had to convert all of the categorical data into numerical data. I did this through a large dictionary in my UCIDataPreprocessing.py file. This would allow the categorical sections to keep their semantic meaning but allow the machine to learn off of them.

For the adult dataset, I used a relatively small MLP since there isn't much data or complexity to the data. My input layer was of size 14, and fed to the first hidden layer of 32. This hidden layer, and all after, had a dropout of 0.2 to limit overfitting. This hidden layer fed to a hidden layer of 16 that had normalization to prevent the model from only guessing the <50k option. This layer then led to the output layer of 1 node, since it is binary classification. 

For the CNN for the adult dataset, I unflattened the data since it was in the form of a pandas dataframe and not a numpy array. I then used a 1d convolutional layer, since the data is 1 dimensional, with a kernel size of 3 and a padding size of 1. This would reduce the dimensionality of the data while adding non-linearity. I then pooled it together to reduce dimensionality even more. I then put it through another 1d convolutional layer and max pooling layer. I then flattened the output to again reduce dimensionality. I then fed it into a MLP of 256 neurons to 32 neurons. This layer had a dropout of 0.2 and fed into a layer of 16 neurons. This layer also had a dropout of 0.2 and fed into the output layer.

My outputs for both were very good! If I wanted to improve them more, I could implement class weights or a loss function that punishes misclassifying the >50K class, as that is where most of my misses came from.


For the Cifar-100 dataset, I was able to use pytorch's built in version of this dataset, removing my need for data preprocessing.

The CNN for the Cifar-100 dataset used two 2D convolutional layers that consisted of 2 filtering layers with 3x3 kernels. These layers created 32 and 64 channels respectively and both used pooling to reduce dimensionality. It then went through an MLP that consisted of a 4096 layer fedding into a layer of 256. This layer of 256 fed into a layer of 128 and the batch was normalized here. This layer of 128 fed into the output layer. I am overall pretty upset with the accuracy that I got for this dataset. I believe a deeper CNN, one using 1x1 CNN layers to reduce dimensionality, and possibly one with more epochs, would drastically improve the accuracy that I got. I wouldn't want to increase the amount of MLP layers as I saw lots of overfitting in my MLP model.

The MLP for the Cifar-100 dataset had a 3072 input layer as that was the amount of inputs in the flattened dataset, which led to a 1024 node hidden layer. This hidden layer went to a 512 node hidden layer, where the batch was normalized. This hidden layer led to a 256 node hidden layer where the batch was normalized. This hidden layer led to the last hidden layer of 128, where the batch was normalized again. This layer then led to the output layer. I'm not sure how to improve this model very much. I already experinced a lot of overfitting as my validation loss was much higher than my training loss, which would suggest that more layers would make this problem worse. I could lessen the amount of nodes in these layers, but it is such a large input pool that it might hurt more than it helps. More dropout may help, but not by a significant amount. More epochs does not seem like it will do much as the loss was hitting a plateau.

For the PCam dataset, I was able to find the original github repository and download the training and testing files from there. I was then able to use the h5py library to read them and split those files into training and testing files.

The CNN for the PCam dataset had 4 convolutional blocks. All of which had a 3x3 kernel size and included batch normalization. They all also included pooling to reduce dinmensionality. The original input layer is size of 3 since RGB is 3 values. Then we go to 32, 64, 128, and 256 layers. I then put those filtered layers through an MLP for classisfication. This has a layer of 9216, the size of the flattened CNN, which feeds into a layer of 512 neurons. This batch, and the next layer of 128, are both normalized. The layer of 128 feeds into the output neuron since this is binary classification.

The MLP for the Pcam had 3 hidden layers since it has a large amount of data, these layers were also quite large and all included normalization. The input layer had 27648 neurons, one for each pixel, and fed into a layer of 2048. This layer then fed into one of size 1024, then 512, then 2256, then the output layer. This should help train for the complexity of the dataset without overfitting. However, this model was extremely overfit. This is definitely due to the large number of neurons. If I were to do this again I would minimize the amount of neurons by a lot!