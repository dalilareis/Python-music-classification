# Python_MusicClassification
## Testing different Machine Learning algorithms to classify music genre

This project proposes a classification model based on 11 different acoustic features related to timbre, melody and harmony. 
The performance of 7 different classifiers is evaluated, as well as 2 methods of dimensionality reduction. The effect of different scalers, transformers and normalizers is also evaluated. 
The proposed system is applied on the recently published dataset, [FMA small](https://github.com/mdeff/fma), that contains 8000 tracks of 30s each, distributed across 8 genres.
Using a feature vector size of 518 dimensions, reduced to 7 vectors by Linear Discriminant Analysis (LDA), we were able to achieve mean accuracy results over 67%, 
evaluated by 10-fold cross validation. These values are on a par with state-of-the-art results for this same task using similar acoustic features extracted from audio content.

### Requirements
The following python libraries are required: **numpy, pandas, matplotlib, sklearn, librosa, pyqt5, scipy, tqdm, pydub**.

These can be easily installed using the **pip** module: `pip install -r requirements.txt`. The modules **pydub** and **librosa** are dependant
of **ffmpeg**, so for the extraction of audio features (e.g., to classify a new audio file), it is necessary to install this framework. 
On Mac, it can be installed using **Homebrew**, `brew install ffmpeg`. On Windows, it can be downloaded through the [official website](https://www.ffmpeg.org/) and the *bin* folder should be added to the system's path.
Detailed instructions can be found [here](http://adaptivesamples.com/how-to-install-ffmpeg-\on-windows/).

A folder containing only a sample of the FMA database was included in this repository for example purposes only. To use this system, the file **tracks.csv** needs to be in the same source folder of this project.
This file is over 200Mb, but it is available in the [fma_metadata.zip](https://github.com/mdeff/fma). 

### GUI
The interface was developped using **pyQt5**. It can be launched with the command `python run.py`. The interface contains 3 tabs:
* **Pre-Processing_Tests:** applies LDA and PCA to the data. It is also possible to see the effect of different normalizers on the data and to test some early classification results of some classifiers. 
* **Model_Results:** tests the different classifiers available, using different normalizers. Possible to select only a subset of features. Saves the model desired to be used to test new data.
* **Classify:** uploads the music file to be classified and shows the resulting estimated **genre**, according to the classifying model saved on the previous tab.

### Scripts available
* `run.py`: launches the user interface.
* `design.py`: contains the design elements of the created GUI. 
* `process.py`: contains the main functions for the pre-processing and training of the classifying models. Linked to the GUI.
* `singleTest.py`: responsible for testing new musics. Includes the sampling of the audio file (30s), feature extraction, and the classification
using the model saved. Linked to the GUI.
* `scalers.py`: additional script (not used by the GUI) to visualize the effects of different normalizers over 2 of the features (average *Tonnetz* and *Zero Crossing Rate*).
* `plot_features.py`: additional script, used to show the features extracted of one example music file from the database.
These plots are divided according to `base` (applied transformations), `chroma`, `spectral` (features related to chroma and spectral representations,
respectively), and `others` (the 2 last features extracted). An example of a command to use this script is: `python plot features.py base`
* `testing.py`: additional script containing different tests. It includes 2 functions, defined by additional arguments:
  1. Compares 3 methods of dimensionality reduction (PCA, NMF, SelectKBest) using linear SVM. **First argument** should be `compare` and the
  **second** specifies the normalizer that should be used, namely:  `standard`, `minMax`, `maxAbs`, `robust`, `quantileUniform`, `quantileNormal`, `L2Norm`
  So an example would be: `python testing.py compare standard`
  2. Finds the best hyperparameters for each classifier.  **First argument** should be `params` and the **second** defines the classifier,
  namely: `RF`, `LR`, `SVM`, `kNN`, `SGD`, `MLP`, `extraTrees`. You can save these parameters directly to a .txt file, using the command: 
  `testing.py params RF >> saveParams.txt`
