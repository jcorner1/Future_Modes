# Classifying Convective Mode in Future Climates
## Introduction

### Data
These data are unique as they follow a couple of Representative Concentration Pathways (RCP(s); both the 4.5 and 8.5 scenarios are modeled in these data), which account for radiative forcing caused by increased levels of greenhouse gases within the atmosphere. RCPs are outlined in the <a href="https://www.ipcc.ch/report/ar5/syr/">Intergovernmental Panel on Climate Change 5th Assessment Report</a> addressing potential atmospheric responses to global climate change. The 4.5 scenario represents a peak in 2040 in carbon emissions and a steady decline till the end. RCP 4.5 is considered an intermediate pathway the most likely to happen in future climates. The 8.5 scenario indicates a worst-case situation with no decline in carbon emissions through 2100.

<p>
    <img src="https://github.com/jcorner1/Future_Modes/blob/main/Plots/All_forcing_agents_CO2_equivalent_concentration.svg.png?raw=true" width="600" height="320" />
</p>

### Objectives
There are many objectives to achieve with this research, and they are as follows:

- Create a machine learning model to classify convective mode.
- View the future shift in convective mode under different climate change scenarios. 
- Use Explainable Artificial Intelligence to understand a machine learning model's interpretation of data.

One of the main goals of this repo is not only to create an open-access format for the code but also to explain it in a manner in which other researchers can utilize these tools for the betterment of the atmospheric sciences (or science in general). This idea is to present all content clearly so as to allow an individual to take this code on classifying convective mode and then use it for their own means (i.e., to classify different types of cloud regimes). Finally, it is important to note that not every bit of code will explained as it would be highly unnecessary to do so, but the most valuable pieces will be presented and broken down so a user can know what needs to be in the code for it to run properly and what can be modified to fit their needs. For all the details/code, please consult the <a href="https://github.com/jcorner1/Future_Modes/tree/main/Code">code directory</a> present in this repo for recreating these methods.

## Convective Mode
Convective Mode is... Convective Mode can be classified in many different manners. However, for the purpose of this work, the labels are an attempt to classify the spectrum between cellular and linear modes. The labels are as such: isolated cell, multiple isolated cells, loosely clustered cells, tightly clustered cells, QLCS, and tropical.​

### Convective Objects
There is an overwhelming need to define an area where "severe weather" occurs in the model data and, therefore generate images and other pertinent data. In the case of this work, a threshold of ..+ UH and ..+ reflectivity values... These thresholds were used with consideration from Gensini and Mote (2015) as well as a sensical understanding of normal storm report numbers. 

#### Thunderstorm Days
Convective objects are sensical for a severe reports-type dataset (tabular), but for understanding the shift in convective mode (a spatial-temporal visualization), the data must match how it is climatologically reported. There are many methods of doing so, but the simplest is in accordance with the thunderstorm day. Thunderstorm days  

## Machine Learning
Machine learning is a powerful, yet complicated technique using advanced statistics to help a model make decisions it is trained to make. 

### Organizing Data
It is important before training any dataset, to divide the data into three categories. The categories are as such: training, validation, and testing. These splits are important to ensure the model is trained and fitted properly. Training is the part of the dataset that the model sees and learns from directly. Validation is the portion of the dataset the model is checked with after each epoch to evaluate the model performance throughout training. The model doesn't truly see this data but rather the performance (accuracy, loss) of its ability to classify the data. Finally, the testing data after training is done to see the overall accuracy. This dataset is kept separated from training to truly check how well the model classifies unseen data. I have used the following metaphor to provide more context and understand why/how the datasets are important to the training process: These datasets are a student learning the material in a single section of a college course. The training dataset is like the notes from the PowerPoint/Slides that help the student (the model in this case) learn. The validation dataset is a weekly quiz that is graded and given back to the student to provide feedback on how to improve. The testing data is the *test* which helps determine how well the student learned the material. There is no defined percent split for each sub-dataset, but the common practice is 70% for training, 10% for validation, and 20% for testing with the understanding that it is okay to change these percentages as needed.

As stated before, training a model without due regard can cause it to overfit/underfit and therefore not accurately classify data in the desired manner. The image below should provide context to the

#### Pickling Data
When using machine learning techniques, it is important to organize data in simple formats as this allows for faster training of the model. A common method of storing numpy arrays (the input for the machine learning model) is using pickle. The below example

``` Python3
# Create the various subsets of data for each of the training datasets
for subset, name in zip([(2007, 2014), (2015, 2016), (2017, 2018)], ("train", "validation", "test")):
    sub_ = classes[(classes.year >= subset[0]) & (classes.year <= subset[1])].copy()
    image_data = []

    # Iterate through each row of the dataframe 
    for rid, row in sub_.iterrows():

        # Open and append each image
        fname = f"{prefix}/{row.radar_time[5:7]}/{row.wfo}/{row.filename}"   
        img = imageio.imread(fname, pilmode='P')
        image_data.append(img)

        # Expand the dimensions and save the labels
        imgs = np.expand_dims(np.array(image_data), axis=3)
        classes_ = sub_['top_class'].values
        class_codes = [np.where(np.array(class_lookup) == classes_[x])[0][0] for x in range(len(classes_))]

        # Dump to a pickle file for later use
        pickle.dump([imgs, class_codes], open("/home/scratch/jcorner1/Thesis/data/nexrad/{}_{}_{}.pkl".format(subset[0], subset[1], name), "wb"))
```

Finally, the notebook used to format these data can be found <a href="https://github.com/jcorner1/Future_Modes/blob/main/Code/Create_Training_Pickles.ipynb">here</a>.

#### Zipping and Unzipping Dataset
Zipping and unzipping can make it easier to transfer files from one computer system to another. Traditional zipping is slower and can corrupt data therefore, the common practice is to use tar to zip a directory with the desired data. The following command is used to zip the directory for fast transfer of data:

``` Python3
tar -zcvf nexrad.tar.gz nexrad
```

This next command is then used to unzip the directory so data can be used:
``` Python3
tar -xf nexrad.tar.gz
```

### Training
Now that the data is prepared and saved in a proper format, the training can begin. This example will use a smaller dataset as it would be impossible to upload the file to this repository. Firstly, the data must be loaded into memory from the pickle files. It is important to note that data is normally standardized or normalized in some manner to 

#### Data Augmentation
Data augmentation is another common practice used in machine learning. Data augmentation is usually required to prevent a model from overfitting (i.e., to keep the model from fitting to the training data too well). Data augmentation usually requires altering the orientation/size of the object in the image. The below images are used to explain the method of data augmentation performed in this study. If the single image from the training dataset is seen by the model during each epoch when compiling it, the model will look at specific details in that image that might not necessarily reflect the desired details the model is supposed to differentiate from.     

<p>
    <img src="https://github.com/jcorner1/Future_Modes/blob/main/Plots/Single_image_noaug.JPG" width="469" height="417" />
</p>

Therefore, to reduce this effect and teach the model to view data in a more general sense (such as how a person might do), augmentation is undergone. Running the code below achieves this effect:

``` Python3
datagen = ImageDataGenerator(rotation_range=55, zoom_range=[0.9,1.0], fill_mode="reflect")
```

and produces this change to the training dataset for every image.

<p>
    <img src="https://github.com/jcorner1/Future_Modes/blob/main/Plots/Data_aug_images.JPG" width="913" height="865" />
</p>

### Classifying Mode

### Explainable Artificial Intelligence
Explainable Artificial Intelligence, or XAI, is a development to solve a prime issue in machine learning, which is understanding how a model makes its predictions (). XAI ... Fiannly, a video with a demonstration of how this code works and what it does can be found <a href="https://www.youtube.com/watch?v=Vx-LF_6fPWo">here</a>. 

#### Backwards Optimization

``` Python3
def _gradient_descent_for_bwo(
        cnn_model_object, loss_tensor, init_function_or_matrices,
        num_iterations, learning_rate):
    """Does gradient descent (the nitty-gritty part) for backwards optimization.

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param loss_tensor: Keras tensor, defining the loss function to be
        minimized.
    :param init_function_or_matrices: Either a function or list of numpy arrays.

    If function, will be used to initialize input matrices.  See
    `create_gaussian_initializer` for an example.

    If list of numpy arrays, these are the input matrices themselves.  Matrices
    should be processed in the exact same way that training data were processed
    (e.g., normalization method).  Matrices must also be in the same order as
    training matrices, and the [q]th matrix in this list must have the same
    shape as the [q]th training matrix.

    :param num_iterations: Number of gradient-descent iterations (number of
        times that the input matrices are adjusted).
    :param learning_rate: Learning rate.  At each iteration, each input value x
        will be decremented by `learning_rate * gradient`, where `gradient` is
        the gradient of the loss function with respect to x.
    :return: list_of_optimized_input_matrices: length-T list of optimized input
        matrices (numpy arrays), where T = number of input tensors to the model.
        If the input arg `init_function_or_matrices` is a list of numpy arrays
        (rather than a function), `list_of_optimized_input_matrices` will have
        the exact same shape, just with different values.
    """

    if isinstance(cnn_model_object.input, list):
        list_of_input_tensors = cnn_model_object.input
    else:
        list_of_input_tensors = [cnn_model_object.input]

    num_input_tensors = len(list_of_input_tensors)
    
    print(loss_tensor)
    print(list_of_input_tensors)
        
    list_of_gradient_tensors = tf.compat.v1.keras.backend.gradients(loss_tensor, list_of_input_tensors)     
    
    for i in range(num_input_tensors):
        list_of_gradient_tensors[i] /= K.maximum(
            K.sqrt(K.mean(list_of_gradient_tensors[i] ** 2)),
            K.epsilon()
        )

    inputs_to_loss_and_gradients = K.function(
        list_of_input_tensors + [K.learning_phase()],
        ([loss_tensor] + list_of_gradient_tensors)
    )

    if isinstance(init_function_or_matrices, list):
        list_of_optimized_input_matrices = copy.deepcopy(
            init_function_or_matrices)
    else:
        list_of_optimized_input_matrices = [None] * num_input_tensors

        for i in range(num_input_tensors):
            these_dimensions = np.array(
                [1] + list_of_input_tensors[i].get_shape().as_list()[1:],
                dtype=int)

            list_of_optimized_input_matrices[i] = init_function_or_matrices(
                these_dimensions)

    for j in range(num_iterations):
        these_outputs = inputs_to_loss_and_gradients(
            list_of_optimized_input_matrices + [0])

        if np.mod(j, 1000) == 0:
            print('Loss after {0:d} of {1:d} iterations: {2:.2e}'.format(
                j, num_iterations, these_outputs[0]))

        for i in range(num_input_tensors):
            list_of_optimized_input_matrices[i] -= (
                these_outputs[i + 1] * learning_rate)

    print('Loss after {0:d} iterations: {1:.2e}'.format(
        num_iterations, these_outputs[0]))
    return list_of_optimized_input_matrices
```

and the code to run the backward optimization and alter the learning rate and the number of iterations for each. 

``` Python3
def bwo_for_class(
        cnn_model_object, target_class, init_function_or_matrices,
        num_iterations=4000,
        learning_rate=0.00000001):
    """Does backwards optimization to maximize probability of target class.

    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param target_class: Synthetic input data will be created to maximize
        probability of this class.
    :param init_function_or_matrices: See doc for `_gradient_descent_for_bwo`.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :return: list_of_optimized_input_matrices: Same.
    """

    target_class = int(np.round(target_class))
    num_iterations = int(np.round(num_iterations))

    assert target_class >= 0
    assert num_iterations > 0
    assert learning_rate > 0.
    assert  learning_rate < 1.

    num_output_neurons = (
        cnn_model_object.layers[-1].output.get_shape().as_list()[-1]
    )

    if num_output_neurons == 1:
        assert target_class <= 1

        if target_class == 1:
            loss_tensor = K.mean(
                (cnn_model_object.layers[-1].output[..., 0] - 1) ** 2
            )
        else:
            loss_tensor = K.mean(
                cnn_model_object.layers[-1].output[..., 0] ** 2
            )
    else:
        assert target_class < num_output_neurons

        loss_tensor = K.mean(
            (cnn_model_object.layers[-1].output[..., target_class] - 1) ** 2
        )

    return _gradient_descent_for_bwo(
        cnn_model_object=cnn_model_object, loss_tensor=loss_tensor,
        init_function_or_matrices=init_function_or_matrices,
        num_iterations=num_iterations, learning_rate=learning_rate)
```
The code for these examples can be found here, with an additional example from the 2023 Unidata Users Workshop found <a href="https://github.com/Unidata/users-workshop-2023/blob/main/3_wednesday/breakout_sessions/XAI/PredictENSO_BO.ipynb">here</a>.

#### Gradient x Input
``` Python3
def get_gradients(inputs, top_pred_idx=None):
    """Computes the gradients of outputs w.r.t input image.
​
    Args:
        inputs: 2D/3D/4D matrix of samples
        top_pred_idx: (optional) Predicted label for the x_data
                      if classification problem. If regression,
                      do not include.
​
    Returns:
        Gradients of the predictions w.r.t img_input
    """
    inputs = tf.cast(inputs, tf.float32)
​
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        
        # Run the forward pass of the layer and record operations
        # on GradientTape.
        preds = model(inputs, training=False)  
        
        # For classification, grab the top class
        if top_pred_idx is not None:
            preds = preds[:, top_pred_idx]
        
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.        
    grads = tape.gradient(preds, inputs)
    return grads
```
The code for these examples can be found here, with an additional example from the 2023 Unidata Users Workshop found <a href="https://github.com/Unidata/users-workshop-2023/blob/main/3_wednesday/breakout_sessions/XAI/PredictENSO_ItG.ipynb">here</a>.

## Conclusion


### References


### Additional Resources
- https://medium.com/latinxinai/how-i-deployed-a-machine-learning-model-for-the-first-time-b82b9ea831e0
- https://github.com/amamalak/XAI-baselines
- https://github.com/djgagne/deepsky
- https://github.com/ai2es/WAF_ML_Tutorial_Part2/tree/main

### Packages
This work was only possible using a few open-source Python packages. These packages and some of the commonly used functions are provided below:

- **Xarray** - A library is normally used for gridded data such as netCDF and grib files. Some of the functions used in this work include:
    - **Coarsen** - Makes data more coarse (i.e., lowers the resolution or increases grid spacing).
    - **Where** - Find values of certain logic (greater than or equal to 40). The function also allows data augmentation by performing basic math on the values meeting the criteria or setting it to a set number.  
- **TensorFlow** - A library that eases the development of workflows when using machine learning. TensorFlow is the most popularly used package in the machine learning industry, and some of the commonly used functions are listed next:
    - **Keras** - Important sublibrary housing different forms of deeper machine learning models. It is most commonly used when working with neural networks. 
- **Pandas** - A library useful when working with tabular data like CSV files.
    - **Dataframe** - A common form of expressing tabular data. Opening a CSV will return a pandas dataframe, but one can also be made from scratch. 
        - **Append** - Adds a row to a dataframe.
- **Numpy** - A library used when creating and changing data within an array. There are many simple functions used when "playing with the data," of which are listed below:
    - **Zeros** - Creates an array made of 0s.
- **SciKit** - A library with
    - **Learn** - 
- **Python** - General functions found in vanilla Python:
    - **Locals** - Function useful in modifying variable names (on the left side of an equal sign; done within a loop) dynamically.
    - **Map** - Function 


## Author Information
Jeremy Corner - M.S. Student &nbsp; &nbsp; &nbsp;  <a href="https://github.com/jcorner1">Github</a> &nbsp; | &nbsp;<a href="https://twitter.com/JcornerWx">Twitter</a> &nbsp; | &nbsp; <a href="mailto:jcorner1@niu.edu">Email</a>


## Committee Members

Alex Haberlie - Advisor  &nbsp; &nbsp; &nbsp;  <a href="http://www.svrimg.org">SVRIMG</a> &nbsp; | &nbsp; <a href="https://github.com/ahaberlie">Github</a> &nbsp; | &nbsp; <a href="https://twitter.com/alexhabe">Twitter</a> &nbsp; | &nbsp; <a href="https://ahaberlie.github.io/">Website</a> &nbsp; | &nbsp; <a href="https://scholar.google.com/citations?user=HvnxYVAAAAAJ">Google Scholar</a> 

Victor Gensini - Member &nbsp; &nbsp; &nbsp;  <a href="https://github.com/vgensini">Github</a> &nbsp; | &nbsp; <a href="https://twitter.com/gensiniwx?lang=en">Twitter</a> &nbsp; | &nbsp; <a href="https://atlas.niu.edu/">Website</a> &nbsp; | &nbsp; <a href="https://scholar.google.com/citations?user=qyLBZwkAAAAJ&hl">Google Scholar</a>

Walker Ashley -  Member &nbsp; &nbsp; &nbsp; <a href="https://twitter.com/WalkerSAshley">Twitter</a> &nbsp; | &nbsp; <a href="https://chubasco.niu.edu/">Website</a> &nbsp; |  &nbsp; <a href="https://scholar.google.com/citations?user=SwhAm7IAAAAJ&hl">Google Scholar</a>

Allison Michealis -  Member &nbsp; &nbsp; &nbsp; <a href="https://twitter.com/WalkerSAshley">Google Scholar</a> &nbsp;

Scott Collis - Collaborator &nbsp; &nbsp; &nbsp; <a href="https://github.com/scollis">Github</a> &nbsp; | &nbsp; <a href="https://twitter.com/Cyclogenesis_au">Twitter</a> &nbsp; | &nbsp; <a href="https://opensky.press/">Website</a> &nbsp; |  &nbsp; <a href="https://scholar.google.com/citations?hl=en&user=eMCDQDIAAAAJ">Google Scholar</a>

## Acknowledgements 

The author and committee members would like to acknowledge the help of Drs. David Gagne, Will Chapman, Kirsten Mayer, and Thomas Martin. We would also like to thank Unidata and Jetstream's computational power and GPU access. Finally, the author thanks his committee members for all their help and his advisor for the many opportunities provided to him to become a better scientist. 
