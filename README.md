# DC-VAE
## Abstract
Due to its unsupervised nature, anomaly detection plays a central role in cybersecurity, in particular on the detection of unknown attacks. A major source of cybersecurity data comes in the form of multivariate time-series (MTS), representing the temporal evolution of multiple, usually correlated measurements. Despite the many approaches available in the literature for time-series anomaly detection, the automatic detection of abnormal events in MTS remains a complex problem. In this paper we introduce DC-VAE, a novel approach to anomaly detection in MTS, leveraging convolutional neural networks (CNNs) and variational auto encoders (VAEs). DC-VAE detects anomalies in time-series data, exploiting temporal information without sacrificing computational and memory resources. In particular, instead of using recursive neural networks, large causal filters, or many layers, DC-VAE relies on Dilated Convolutions (DC) to capture long and short term phenomena in the data, avoiding complex and less-efficient deep architectures, simplifying learning. We evaluate DC-VAE on the detection of anomalies on a large-scale, multi-dimensional network monitoring dataset collected at an operational mobile Internet Service Provider (ISP), where anomalous events were manually labeled during a time span of 7-months, at a five-minutes granularity. Results show the main properties and advantages introduced by VAEs for time-series anomaly detection, as well as the out-performance of dilated convolutions as compared to standard VAEs for time-series modeling.

<img src="https://user-images.githubusercontent.com/68783507/167413824-7687137a-1480-4640-8c6a-733676f2847e.gif" alt="drawing" width="50%" height="50%"/>

## Run the code 
### Environment setup
The exact specification of our environment is provided in the file environment.yml.
```
conda env create -f environment.yml
```

The command above will create an environment dc-vae with all the required dependencies. 
After you create the environment, activate it.
```
conda activate dc-vae
```

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

