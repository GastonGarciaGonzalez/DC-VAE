# DC-VAE

### _One Model to Find them All – Deep Learning for Multivariate Time-Series Anomaly Detection in Mobile Network Data_

In the following repository we intend to make available the python code used in the experimental part of our work. We welcome any comments or suggestions that help improve the repository.

## Abstract
The automatic detection of anomalies in communication networks plays a central role in network management.
Despite the many attempts and approaches for anomaly detection explored in the past, the detection of rare events in
multidimensional network data streams still represents a complex
to tackle problem. Network monitoring data generally consists
of hundreds of counters periodically collected in the form of
time-series, resulting in a complex-to-analyze multivariate timeseries (MTS) process. Traditional time-series anomaly detection
methods target univariate time-series analysis, which makes the
multivariate analysis cumbersome and prohibitively complex
when dealing with MTS data. In this paper we introduce DC-VAE, a novel approach to anomaly detection in MTS data,
leveraging convolutional neural networks (CNNs) and variational
auto encoders (VAEs). DC-VAE detects anomalies in MTS data
through a single model, exploiting temporal information without
sacrificing computational and memory resources. In particular,
instead of using recursive neural networks, large causal filters,
or many layers, DC-VAE relies on Dilated Convolutions (DC) to
capture long and short term phenomena in the data, avoiding
complex and less-efficient deep architectures, thus simplifying
learning. We evaluate DC-VAE on the detection of anomalies
in the TELCO dataset, a large-scale, multi-dimensional network
monitoring dataset collected at an operational mobile Internet
Service Provider (ISP), where anomalous events were manually
labeled by experts during a time span of seven-months, at a fiveminutes granularity. Using this dataset, we benchmark DC-VAE
against a broad set of traditional time-series anomaly detectors
coming from the signal processing and machine learning domains.
We also evaluate DC-VAE in open, publicly available datasets,
comparing its performance against other multivariate anomaly
detectors based on deep learning generative models. Results show
the main properties and advantages introduced by VAEs for MTS
anomaly detection, as well as the out-performance of dilated
convolutions as compared to standard VAEs and traditional
univariate approaches for time-series modeling. For the sake of
reproducibility and as an additional contribution, we make the
TELCO dataset publicly available to the community, and openly
release the code implementing DC-VAE.

<img src="https://user-images.githubusercontent.com/68783507/168036163-c70fca64-f12e-4053-a05b-5b8fbd9c8dd6.png" alt="drawing" width="60%" height="60%"/><img src="https://user-images.githubusercontent.com/68783507/167413824-7687137a-1480-4640-8c6a-733676f2847e.gif" alt="drawing" width="40%" height="40%"/>

## Run the code 
- python >= 3.6
- tensorflow >= 2.4

We recommend using [Anaconda](https://www.anaconda.com/). 

Open Anaconda Prompt to execute the following lines

### Environment 
To create an environment with the requirements for the detector to work, execute the following line.
```
conda create --name <env> --file requirements.txt
```
The library to run the custom metrics specified in the paper can only be installed with pip. If you want to evaluate your results with these metrics, install the [_prts_](https://github.com/CompML/PRTS) library.

```
pip install prts 
```

After you create the environment, activate it.
```
conda activate <env>
```

### Data
The data format must be a table where the first column corresponds to the timestamp. They must be saved in a .csv file.
You could try with our own dataset [TELCO](https://iie.fing.edu.uy/investigacion/grupos/anomalias/en/telco-dataset-2/downloads/) if you want.

### Settings
All the values and hyperparameters necessary to create a DC-VAE model must be specified in a .txt file. An example of this can be seen in the _settings_ folder.

### Training
To train a model you must execute the *train.py* script and indicate the data train and settings files.

```
python train.py data_train.csv settings\model_settings.txt
```

### Alpha definition
If the training set contains labels, the alpha values can be adjusted in order to maximize the value of F1 in detections. Execute the *alpha_definition.py* script and indicate the train data and labels files and the settings file.

```
python alpha_definition.py data_train.csv labels_train.csv settings\model_settings.txt
```

### Testing
To test the trained model execute the *test.py* script and indicate the test data and settings files.

```
python test.py data_test.csv settings\model_settings.txt
```


## Cite
```

```

## Acknowledgment
This work has been partially supported by the ANII-FMV project with reference FMV-1-2019-1-155850 Anomaly Detection with Continual and Streaming Machine Learning on Big Data Telecommunications Networks, by Telefonica, and by the Austrian FFG ICT-of-the-Future project _DynAISEC – Adaptive AI/ML for Dynamic Cybersecurity Systems_. Gaston García González a was supported by the ANII scholarship POS-FMV-2020-1-1009239, and by CSIC, under program Movilidad e Intercambios Academicos 2022.

