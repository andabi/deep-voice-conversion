multi modal을 학습하는 것은 어렵다.
적절한 규제를 하지 않으면 mu, pi 등이 같은 값으로 모여버리면서 국소점에 빠진다.
mu를 몰리지 않게 강제화하는 loss term 고려할 수 있다.
loss가 한계에 다다르면 더 이상 떨어지지 않음.

variance도 같이 학습해야 한다.
처음에 1로 고정하고 mu만 학습했더니 분포의 표현력이 좋지 않아 높은 likelihood를 얻을 수 없었다.

log(prob + e)에서 e의 값의 크기가 학습되는 likelihood에 영향을 미친다. 
더 작은 e일수록 더 큰 likelihood를 얻는다. e를 없애고도 학습할 수 있는 방법 도입해야 함.

logsumexp

pixel cnn++을 참고하자