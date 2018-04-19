multi modal을 학습하는 것은 어렵다.
적절한 규제를 하지 않으면 mu, pi 등이 같은 값으로 모여버리면서 국소점에 빠진다.
mu를 몰리지 않게 강제화하는 loss term 고려할 수 있다.
loss가 한계에 다다르면 더 이상 떨어지지 않음.

variance도 같이 학습해야 한다.
처음에 1로 고정하고 mu만 학습했더니 분포의 표현력이 좋지 않아 높은 likelihood를 얻을 수 없었다.

inference시 dominant dist.를 고름. 따라서 학습시에도 여러 분포 중에 나온 값 중 최대를 maximize하는 식으로 함.
mixture를 하나의 분포로 보기보다는 여러 개의 서로 다른 single dist.가 있는 것으로 보는게 맞음.
* logsumexp

edge 케이스

pi softmax 앞에 normalization 중요하다.

---
documentation
- why mol?
 - the importance on multi modal modeling in generation task.
- experiments on single and multi modal
 - step-by-step explanation on code
 - tips
 - result