# Please give us feedback to help us improve the exercises. You get 1 bonus point for submitting feedback.

## Major Problems?
For some reasons incorrect code for L1 and L2 regularization firstly run successfully through the tests. 
Incorrect code that I have implemented for L1 forward method:
```python
return self.lambd * np.sum(np.abs([param.data for param in self.params]))
```
Code after (a correct one, I have realized that I have made a mistake when I was running `train_models.py` module):
```python
return self.lambd * np.sum([np.sum(np.abs(param.data)) for param in self.params])
```
I was not doing the necessary summation, but tests did not catch this error. 

## Helpful?
Comments, tests and lectures are very helpful as always.


## Duration (hours)?

_Please make a list where every student in your group puts in the hours they used to do the complete exercise_
_Example: [5.5, 4, 7], if one of you took 5 and a half hours, one person 4 and the other 7. The order does not matter._
_This feedback will help us analyze which exercise sheets are too time-intensive._

[4.5, 6, 6]

## Other feedback?



