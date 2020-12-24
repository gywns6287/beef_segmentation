# Sex estimation from pig carcasses

## Object

The sex of pig carcasses is one of the important factors that have an influence on pork price.  Currently,  trained human inspect all pig carcasses one by one, taking time and effort. Therefore, a more efficient method to determining the sex of pig carcasses should be developed.

## Model & Code

our models were implemented by **tensorflow 2.3** and **keras**

#### Model summary
1. This model was based on ResNet50.
2. Class Activation Map was printed together with output results.
3. Model code can be found in the `CAM_LHJ.py`.

#### Example with training
```
python solution.py --test data\\test --train data\\train --out .
```
#### Example without training
```
python solution.py --test data\\test --out .
```
The pre-trained weight must exist as `weight.h5` in the path where `solution.py` is located.

## Datasets & pre-trained weight

The `data` folder contains only simple sample images and does not contain the data used for model training.  All the data-set used in this study was provided by Artificial Intelligence Convergence Research Center(Chungnam National University)). Request for academic purposes can be made to gywns6298@naver.com.

pre-trained weight can be downloaded at https://drive.google.com/drive/folders/1jcsroiExir9e4PKU6kFVmHBAZLdaAMx5?usp=sharing


## Model output

1. `pred.sol`: estimation results for test image set.
2. `summary.txt`: summary of test results (accuracy, F1-score).
3. **CAM**

  Female
  
  ![Female](https://user-images.githubusercontent.com/71325306/94219528-0240e600-ff22-11ea-8bf5-a708fe9f17ae.png)
  
  Male
  
  ![Male](https://user-images.githubusercontent.com/71325306/94219530-040aa980-ff22-11ea-96df-b3145fade5ee.png)
  
  Boar
  
  ![Boar](https://user-images.githubusercontent.com/71325306/94219434-cd349380-ff21-11ea-9f99-e1b91adda17b.png)
  
## Test results
Model was trained by 5,909 pig carcass images,
and test set contain 1,386 images.

|Acc  |F1 score|
|-----|--------|
|0.993|0.990   |

