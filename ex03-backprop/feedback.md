# Please give us feedback to help us improve the exercises. You get 1 bonus point for submitting feedback.

## Major Problems?
Most of the time I take 2-4 minutes to understand what is wanted.  Then I do some stupid typo and sometimes it takes up to 
two hours to find it, as I reread the slides and do the calculation from hand or draw the network. This really 
frustrating, because I know I have it right, but the typo  messes it up... The issue this time was in Linear.backward as
I returned:\
`grad @ self.W.grad.transpose()`  # dL/dx = (dL/dy) * w.T
\
instead of:
\
`grad @ self.W.data.transpose()`  # dL/dx = (dL/dy) * w.T
\
I mean I even wrote it next to it. Most of the time I just do something else and come back, which is difficult for those
kind of tasks, as they build on each other. Learning by doing I guess and I also repeat the content of the lecture like
this, so at least something comes from it. 
------------------------------------------------------------------------------------------------------------------------
We do not understand why there is a += done on the gradients of the weights and biases in lines 63 and 64 of network.py. 

## Helpful?

In the past exercises I totally forgot to mention how helpful the test are!!! The comments were helpful as always.
Comments and links to the resources are very useful.

## Duration (hours)?

_Please make a list where every student in your group puts in the hours they used to do the complete exercise_
_Example: [5.5, 4, 7], if one of you took 5 and a half hours, one person 4 and the other 7. The order does not matter._
_This feedback will help us analyze which exercise sheets are too time-intensive._

[8, 7, 10]
## Other feedback?



