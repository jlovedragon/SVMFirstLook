#SVM

using scikit-learn do execrises about SVM
environment:Ubuntu14.04 sublime2 python2.7.6

##  Classification contrast :
* Linear model: compare with Perceptron
    * plot_linear_model.py
    * plot_perceptron_test1.py (different coef_init and intercept_init)
    * plot_perceptron_test2.py (different order)

* Non-linear model: To explain the performance of the svm is sensitive to the different kernel:
    * plot_svm_kernel.py (come from http://scikit-learn.org/)
    * plot_balancescale_test.py


## Regression contrast :
* (RBF, linear, poly): plot_svm_regression.py (come from http://scikit-learn.org/)

## 	How to choose parameters C and gamma in linear and RBF kernel ?
* plot_linear_parameters_different_c.py
* plot_linear_parameters_different_c_one_noise.py
* plot_rbf_parameters_different_c.py
* plot_rbf_parameters_different_gamma.py

****************************************************************************

## DataSets :
* iris
	* Number of Instances: 150 (50 in each of three classes)
	* Number of Attributes: 4 numeric, predictive attributes and the class

	Summary Statistics:
	         	   Min  Max   Mean    SD   Class Correlation
   	sepal length = 4.3  7.9   5.84  0.83    0.7826
    sepal width  = 2.0  4.4   3.05  0.43   -0.4194
   	petal length = 1.0  6.9   3.76  1.76    0.9490  (high!)
    petal width  = 0.1  2.5   1.20  0.76    0.9565  (high!)

*	balance scale (polynomial)
	*	Number of Instances: 625 (49 balanced, 288 left, 288 right)
	*	Number of Attributes: 4 (numeric) + class name = 5

	Later, I have already discarded the datas of 49 balanced to hold the same weight of the dataset.


##关于作者

```javascript
    nickName = "Quentin_Hsu"
    email    = "jlove.dragon@gmail.com"
```
