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

```
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

Finally, the notebook used to format these data can be found **[[[[here]]]]**. 

#### Zipping and Unzipping Dataset
Zipping and unzipping can make it easier to transfer files from one computer system to another. Traditional zipping is slower and can corrupt data therefore, the common practice is to use tar to zip a directory with the desired data. The following command is used to zip the directory for fast transfer of data:

```
tar -zcvf nexrad.tar.gz nexrad
```

This next command is then used to unzip the directory so data can be used:
```
tar -xf nexrad.tar.gz
```

### Training
Now that the data is prepared and saved in a proper format, the training can begin. This example will use a smaller dataset as it would be impossible to upload the file to this repository. Firstly, the data must be loaded into memory from the pickle files. It is important to note that data is normally standardized or normalized in some manner to 

#### Data Augmentation


### Classifying Mode

### Explainable Artificial Intelligence
Explainable Artificial Intelligence, or XAI, is a development to solve a prime issue in machine learning which is understanding how a model makes its predictions (). XAI ...

#### Backwards Optimization

#### Gradient x Input

## Conclusion

### References

### Additional Resources
- https://medium.com/latinxinai/how-i-deployed-a-machine-learning-model-for-the-first-time-b82b9ea831e0
- https://github.com/amamalak/XAI-baselines
- https://github.com/djgagne/deepsky
- https://github.com/ai2es/WAF_ML_Tutorial_Part2/tree/main

### Packages


## Author Information
Jeremy Corner - M.S. Student &nbsp; &nbsp; &nbsp;  <a href="https://github.com/jcorner1">Github</a> &nbsp; | &nbsp;<a href="https://twitter.com/JcornerWx">Twitter</a> &nbsp; | &nbsp; <a href="mailto:jcorner1@niu.edu">Email</a>


## Committee Members

Alex Haberlie - Advisor  &nbsp; &nbsp; &nbsp;  <a href="http://www.svrimg.org">SVRIMG</a> &nbsp; | &nbsp; <a href="https://github.com/ahaberlie">Github</a> &nbsp; | &nbsp; <a href="https://twitter.com/alexhabe">Twitter</a> &nbsp; | &nbsp; <a href="https://ahaberlie.github.io/">Website</a> &nbsp; | &nbsp; <a href="https://scholar.google.com/citations?user=HvnxYVAAAAAJ">Google Scholar</a> 

Victor Gensini - Member &nbsp; &nbsp; &nbsp;  <a href="https://github.com/vgensini">Github</a> &nbsp; | &nbsp; <a href="https://twitter.com/gensiniwx?lang=en">Twitter</a> &nbsp; | &nbsp; <a href="https://atlas.niu.edu/">Website</a> &nbsp; | &nbsp; <a href="https://scholar.google.com/citations?user=qyLBZwkAAAAJ&hl">Google Scholar</a>

Walker Ashley -  Member &nbsp; &nbsp; &nbsp; <a href="https://twitter.com/WalkerSAshley">Twitter</a> &nbsp; | &nbsp; <a href="https://chubasco.niu.edu/">Website</a> &nbsp; |  &nbsp; <a href="https://scholar.google.com/citations?user=SwhAm7IAAAAJ&hl">Google Scholar</a>

Scott Collis - Collaborator &nbsp; &nbsp; &nbsp; <a href="https://github.com/scollis">Github</a> &nbsp; | &nbsp; <a href="https://twitter.com/Cyclogenesis_au">Twitter</a> &nbsp; | &nbsp; <a href="https://opensky.press/">Website</a> &nbsp; |  &nbsp; <a href="https://scholar.google.com/citations?hl=en&user=eMCDQDIAAAAJ">Google Scholar</a>

## Acknowledgements 

The author and committee members would like to acknowledge the... We would also like to thank Unidata and Jetstream's computational power and GPU access. Finally, the author thanks his committee members for all their help and his advisor for the many opportunities provided to him to become a better scientist. 
