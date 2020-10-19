title: # Intermediate Machine Learning with scikit-learn: Evaluation, Calibration, and Inspection
use_katex: True
class: title-slide

# Intermediate Machine Learning with scikit-learn
## Evaluation, Calibration, and Inspection

![](images/scikit-learn-logo-notext.png)

.larger[Thomas J. Fan]<br>
@thomasjpfan<br>
<a href="https://www.github.com/thomasjpfan" target="_blank"><span class="icon icon-github icon-left"></span></a>
<a href="https://www.twitter.com/thomasjpfan" target="_blank"><span class="icon icon-twitter"></span></a>
<a class="this-talk-link", href="https://github.com/thomasjpfan/ml-workshop-intermediate-2-of-2" target="_blank">
This workshop on Github: github.com/thomasjpfan/ml-workshop-intermediate-2-of-2</a>

---

name: table-of-contents
class: middle, larger

# Table of Contents
.g[
.g-6[
1. [Model Evaluation](#evaluation)
1. [Model Calibration](#calibration)
1. [Model Inspection](#inspection)
]
.g-6.g-center[
![](images/scikit-learn-logo-notext.png)
]
]

---

name: evaluation
class: chapter-slide

# 1. Model Evaluation

.footnote[
[Back to Table of Contents](#table-of-contents)
]

???

- classification models
- confusion matrix
- positive and negative
- accuracy?
- precision, recall, f-score
- averaging strategies
- balanced accuracy
- Goal setting
- precision-recall curve
- roc curve
- compare models
- average precision
- threshold metrics vs ranking
- multi-class
- notebook

---

- Regression metrics
- cross_val_score

- scoring interface

---

name: calibration
class: chapter-slide

# 2. Model Calibration

.footnote[
[Back to Table of Contents](#table-of-contents)
]

???

- What is calibration
- Calibrated vs non calibrated
- calibration curves
- calibration in scikit-learn


---

name: inspection
class: chapter-slide

# 3. Model Inspection

.footnote[
[Back to Table of Contents](#table-of-contents)
]

???

- linear models for classification
- linear models for regression
- tree based models
- gradient boosting
- permutation importance
- partial dependence

---

class: title-slide, left

# Closing

.g.g-middle[
.g-7[
![:scale 30%](images/scikit-learn-logo-notext.png)
1. [Model Evaluation](#evaluation)
1. [Model Calibration](#calibration)
1. [Model Inspection](#inspection)
]
.g-5.center[
<br>
.larger[Thomas J. Fan]<br>
@thomasjpfan<br>
<a href="https://www.github.com/thomasjpfan" target="_blank"><span class="icon icon-github icon-left"></span></a>
<a href="https://www.twitter.com/thomasjpfan" target="_blank"><span class="icon icon-twitter"></span></a>
<a class="this-talk-link", href="https://github.com/thomasjpfan/ml-workshop-intermediate-2-of-2" target="_blank">
This workshop on Github: github.com/thomasjpfan/ml-workshop-intermediate-2-of-2</a>
]
]
