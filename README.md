# DC-VAE
[DC-VAE, Fine-grained Anomaly Detection in Multivariate Time-Series with
Dilated Convolutions and Variational Auto Encoders
](https://www.colibri.udelar.edu.uy/jspui/bitstream/20.500.12008/31392/1/GMFGAC22.pdf)
## Abstract
Due to its unsupervised nature, anomaly detection plays a central role in cybersecurity, in particular on the detection of unknown attacks. A major source of cybersecurity data comes in the form of multivariate time-series (MTS), representing the temporal evolution of multiple, usually correlated measurements. Despite the many approaches available in the literature for time-series anomaly detection, the automatic detection of abnormal events in MTS remains a complex problem. In this paper we introduce DC-VAE, a novel approach to anomaly detection in MTS, leveraging convolutional neural networks (CNNs) and variational auto encoders (VAEs). DC-VAE detects anomalies in time-series data, exploiting temporal information without sacrificing computational and memory resources. In particular, instead of using recursive neural networks, large causal filters, or many layers, DC-VAE relies on Dilated Convolutions (DC) to capture long and short term phenomena in the data, avoiding complex and less-efficient deep architectures, simplifying learning. We evaluate DC-VAE on the detection of anomalies on a large-scale, multi-dimensional network monitoring dataset collected at an operational mobile Internet Service Provider (ISP), where anomalous events were manually labeled during a time span of 7-months, at a five-minutes granularity. Results show the main properties and advantages introduced by VAEs for time-series anomaly detection, as well as the out-performance of dilated convolutions as compared to standard VAEs for time-series modeling.

<img src="https://user-images.githubusercontent.com/68783507/168036163-c70fca64-f12e-4053-a05b-5b8fbd9c8dd6.png" alt="drawing" width="50%" height="50%"/><img src="https://user-images.githubusercontent.com/68783507/167413824-7687137a-1480-4640-8c6a-733676f2847e.gif" alt="drawing" width="40%" height="40%"/>

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
@inproceedings{garcia2022dc,
  title={DC-VAE, Fine-grained anomaly detection in multivariate time-series with dilated convolutions and variational auto encoders},
  author={Garc{\'\i}a Gonz{\'a}lez, Gast{\'o}n and Mart{\'\i}nez Tagliafico, Sergio and Fern{\'a}ndez, Alicia and G{\'o}mez, Gabriel and Acu{\~n}a, Jos{\'e} and Casas, Pedro},
  booktitle={7th International Workshop on Traffic Measurements for Cybersecurity (WTMC 2022), Genoa, Italy, jun 6 2022, pp 1-7},
  year={2022},
  organization={IEEE}
}
```

## Acknowledgment
This work has been partially supported by the ANII-FMV project with reference FMV-1-2019-1-155850 Anomaly Detection with Continual and Streaming Machine Learning on Big Data Telecommunications Networks, by Telefonica, and by the Austrian FFG ICT-of-the-Future project _DynAISEC – Adaptive AI/ML for Dynamic Cybersecurity Systems_. Gaston García González a was supported by the ANII scholarship POS-FMV-2020-1-1009239, and by CSIC, under program Movilidad e Intercambios Academicos 2022.

