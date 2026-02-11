# DSDA385_HW1
Machine Learning Capstone Homework 1

Dataset     Architecture        Accuracy        F1          Time to Train (s)                      Notes
Adult           MLP              0.7900        0.4142       6.296967506408691                      Using class weights or a custom loss  
                                                                                                   function could help the accuracy
Adult           CNN              0.7947        0.3112       15.32059931755066

CIFAR-100       MLP              
CIFAR-10        CNN              0.4522       0.4370        492.013863325119

PCam            MLP
PCam            CNN


The objective of this homework assignment was to get comfortable in the pytorch library, to learn the differences in certain types of machine learning models, and to learn how these different models perform on different types of data. 
Each of my MLPs and CNNs were split into different files since I wanted them to be easily accessible and readable.

All layers with activation functions had ReLU functions as it is widely regarded as the best general activation function. The loss I used was binary cross entropy or cross entropy depending on if it was binary classification or not as that is what I am used to working with. All of my models will include an inverse pyramid style of neuron sizes, ex. 128, 64, 32, 16. This allows for more depth while not overfitting the model.

The adult data preprocessing was super interesting since it was a split of numerical and categorical data. Since basic machine learning models, like mine, cannot take categorical data, I had to convert all of the categorical data into numerical data. I did this through a large dictionary in my UCIDataPreprocessing.py file. This would allow the categorical sections to keep their semantic meaning but allow the machine to learn off of them.

For the adult dataset, I used a relatively small MLP since there isn't much data or complexity to the data. My input layer was of size 14, and fed to the first hidden layer of 32. This hidden layer, and all after, had a dropout of 0.2 to limit overfitting. This hidden layer fed to a hidden layer of 16 that had normalization to prevent the model from only guessing the <50k option. This layer then led to the output layer of 1 node, since it is binary classification. 

For the CNN for the adult dataset, I unflattened the data since it was in the form of a pandas dataframe and not a numpy array. I then used a 1d convolutional layer, since the data is 1 dimensional, with a kernel size of 3 and a padding size of 1. This would reduce the dimensionality of the data while adding non-linearity. I then pooled it together to reduce dimensionality even more. I then put it through another 1d convolutional layer and max pooling layer. I then flattened the output to again reduce dimensionality. I then fed it into a MLP of 256 neurons to 32 neurons. This layer had a dropout of 0.2 and fed into a layer of 16 neurons. This layer also had a dropout of 0.2 and fed into the output layer.

My outputs for both were very good! If I wanted to improve them more, I could implement class weights or a loss function that punishes misclassifying the >50K class, as that is where most of my misses came from.


For the Cifar-100 dataset, I was able to use pytorch's built in version of this dataset, removing my need for data preprocessing.



