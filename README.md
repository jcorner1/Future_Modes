# Classifying Convective Mode in Future Climates
## Introduction

### Data
These data are unique as they follow a couple of Representative Concentration Pathways (RCP(s); both the 4.5 and 8.5 scenarios are modeled in these data), which account for radiative forcing caused by increased levels of greenhouse gases within the atmosphere. RCPs are outlined in the <a href="https://www.ipcc.ch/report/ar5/syr/">Intergovernmental Panel on Climate Change 5th Assessment Report</a> addressing potential atmospheric responses to global climate change. The 4.5 scenario represents a peak in 2040 in carbon emissions and a steady decline till the end. RCP 4.5 is considered an intermediate pathway the most likely to happen in future climates. The 8.5 scenario indicates a worst-case situation with no decline in carbon emissions through 2100.

<p>
    <img src="https://github.com/jcorner1/Future_Modes/blob/main/Plots/All_forcing_agents_CO2_equivalent_concentration.svg.png?raw=true" width="600" height="320" />
</p>

### Objectives
There are many objectives to achieve with this research, and they are as follows:

- View the future shift in convective mode under different climate change scenarios. 
- Use Explainable Artificial Intelligence to understand a machine learning model's interpretation of input.

One of the main goals of this repo is not only to create an open-access format for the code but also to explain it in a manner in which other researchers can utilize these tools for the betterment of the atmospheric sciences. This idea is to present all content clearly so as to allow an individual to take this code on classifying convective mode and use it to classify different types of clouds. Finally, it is important to note that not every bit of code will explained as it would be highly unnecessary to do so, but the most valuable pieces will be presented and broken down so a user can know what needs to be in the code for it to run properly and what can be modified to fit their needs. For all details, please consult the code present in this repo for recreating these methods.

## Convective Mode

### Convective Objects

#### Thunderstorm Days

#### Coarsening Data

## Machine Learning

### Organizing Data

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
Explainable Artificial Intelligence, or XAI, is a development to solve a prime issue in machine learning which is understanding how a model makes its predictions (). XAI ...

#### Backwards Optimization


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
    - **Keras** - Important sublibrary housing different forms of deeper machine learning models. Most commonly used when working with neural networks. 
- **Pandas** - A library useful when working with tabular data like CSV files.
    - **Dataframe** - A common form of expressing tabular data. Opening a CSV will return a pandas dataframe, but one can also be made from scratch. 
        - **Append** - Adds a row to a dataframe.
- **Numpy** - A library used when creating and changing data within an array. There are many simple functions used when "playing with the data," of which are listed below:
    - **Zeros** - Creates an array made of 0s.
- **Python** - General functions found in the Python library:
    - **Locals** - Function useful in modifying variables dynamically.


## Author Information
Jeremy Corner - M.S. Student &nbsp; &nbsp; &nbsp;  <a href="https://github.com/jcorner1">Github</a> &nbsp; | &nbsp;<a href="https://twitter.com/JcornerWx">Twitter</a> &nbsp; | &nbsp; <a href="mailto:jcorner1@niu.edu">Email</a>


## Committee Members

Alex Haberlie - Advisor  &nbsp; &nbsp; &nbsp;  <a href="http://www.svrimg.org">SVRIMG</a> &nbsp; | &nbsp; <a href="https://github.com/ahaberlie">Github</a> &nbsp; | &nbsp; <a href="https://twitter.com/alexhabe">Twitter</a> &nbsp; | &nbsp; <a href="https://ahaberlie.github.io/">Website</a> &nbsp; | &nbsp; <a href="https://scholar.google.com/citations?user=HvnxYVAAAAAJ">Google Scholar</a> 

Victor Gensini - Member &nbsp; &nbsp; &nbsp;  <a href="https://github.com/vgensini">Github</a> &nbsp; | &nbsp; <a href="https://twitter.com/gensiniwx?lang=en">Twitter</a> &nbsp; | &nbsp; <a href="https://atlas.niu.edu/">Website</a> &nbsp; | &nbsp; <a href="https://scholar.google.com/citations?user=qyLBZwkAAAAJ&hl">Google Scholar</a>

Walker Ashley -  Member &nbsp; &nbsp; &nbsp; <a href="https://twitter.com/WalkerSAshley">Twitter</a> &nbsp; | &nbsp; <a href="https://chubasco.niu.edu/">Website</a> &nbsp; |  &nbsp; <a href="https://scholar.google.com/citations?user=SwhAm7IAAAAJ&hl">Google Scholar</a>

Scott Collis - Collaborator &nbsp; &nbsp; &nbsp; <a href="https://github.com/scollis">Github</a> &nbsp; | &nbsp; <a href="https://twitter.com/Cyclogenesis_au">Twitter</a> &nbsp; | &nbsp; <a href="https://opensky.press/">Website</a> &nbsp; |  &nbsp; <a href="https://scholar.google.com/citations?hl=en&user=eMCDQDIAAAAJ">Google Scholar</a>

## Acknowledgements 

The author and committee members would like to acknowledge the... We would also like to thank Unidata and Jetstream's computational power and GPU access. Finally, the author thanks his committee members for all their help and his advisor for the many opportunities provided to him to become a better scientist. 
