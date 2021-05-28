from aloha import Aloha
import matplotlib.pyplot as plt
n=50
q=[i*1.0/100 for i in range(5, 100, 5)]
p=0.4
e=20000
accuracies=[]*len(q)
for i in range(len(q)):
    x = Aloha(n, p, q[i], e)
    x.simulate()
    accuracies[i]=(x.result.count(True)*1.0/e)
A=plt.figure(1)
plt.xlabel("Message generation probability")
plt.ylabel("Accuracy")
A.show()
