 -- This is an ongoing project and not the final result. --

## Improvement of data imbalance problem in human motion recognition
<br>
Using the ACGAN idear feature generator
<br><br>
<img src="https://github.com/seungjae-won/feature_generator__human_motion/blob/master/figure/model_figure.PNG" align="left" height="300" width="500" >

<br><br><br><br><br><br><br><br><br><br><br><br>


<h3>Abstract</h3>
Class imbalance problem of data degrades classification performance. The same goes for the field of human motion recognition. Through a feature generator using ACGAN idea, I'm trying to improve the data imbalance problem in the field of human motion recognition. To compare performance through over-sampling and weight balancing, which are used to solve traditional data imbalance problems. "MSRC-12" provided by Microsoft is used as the dataset (Other datasets will be used in the future)


### Dataset
Download-dataset : [MSRC-12 - download](https://www.microsoft.com/en-us/download/details.aspx?id=52283) <br>
Dataset document : [Reference document](https://nanopdf.com/download/this-document-microsoft-research_pdf)

There are a total of 12 classes, approximately 50 sequences for each class were used for training and 10 sequences were used for the test. Data sampling is used to solve the inherent problem of human motion recognition, the difference in sequence length for each motion.



### Method
<img src="https://github.com/seungjae-won/feature_generator__human_motion/blob/master/figure/proposed_method.PNG" align="left" height="300" width="500" >
<br><br><br><br><br><br><br><br><br><br><br><br><br><br>


### Results
| Imbalane rate | Original LSTM | Over-sampling(smoth) | Weight-balancing | Proposed method |
| :-------------: |------------:|---------:|---------:| --:|
| 2 : 1      | 85.00 % | 41.67 % | 12.50 % | 21.67 % |
| 3 : 1      | 79.17 % | 35.00 % | 11.67 % | 74.17 % |
| 5 : 1      | 71.67 % | 24.17 % | 10.83 % | 48.33 % |
| 10 : 1      | 65.00 % | 24.17 % | 15.00 % | 41.67 % |


### Discussion
1. Features that have passed LSTM may not be suitable
2. The performance of weight balancing is remakably low.
3. As oversampling also seems to have degraded performace, consider the oversampling technique of sequence data.
