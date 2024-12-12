# Behavioral data analysis code for Point-SPV


This folder contains the code executed on the paper below for analyzing the behavioral data collected in the behavioral experiment.


```
The paper citation info will be provided after publication
```



The code for the gaze-contingent experiment can be found here:
[SPV-Gaze-Contingency](https://github.com/LEO-UMCG/SPV-Gaze-Contingency)


First set the variable *METHOD* in the `config.py` to either *GSPV* for Point-SPV or *ED* for edge detection and run `data_process.py` to preprocess the data from the raw collected data.

The run `analyze.py` to obtain the figures.


Required dependencies:

```
pickle
numpy
sklearn
matplotlib
```