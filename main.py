import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def Calc_P(x1, x2, p_w, p_d, A):
    p_est = x1 * torch.exp(A[0] * (A[1] * x2 / (A[0] * x1 + A[1] * x2)) ** 2) * p_w + x2 * torch.exp(A[1] * (A[0] * x1 / (A[0] * x1 + A[1] * x2)) ** 2) * p_d
    return p_est
def Plot_Results(p, p_est, x1):
    x1 = x1.detach().numpy()
    p = p.detach().numpy()
    p_est = p_est.detach().numpy()

    plt.figure()
    plt.plot(x1, p, label='Exact P')
    plt.plot(x1, p_est, label='Approx. P')
    plt.xlim([0,1])
    plt.xlabel('x1')
    plt.ylabel('P')
    plt.legend()
    plt.savefig('P1-3.jpg', dpi=300, bbox_inches='tight')
    plt.show()

a = np.array([[8.07131, 1730.63, 233.426], [7.43155, 1554.679, 240.337]])
x1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
x2 = 1 - x1
x1 = torch.tensor(x1, requires_grad=False, dtype=torch.float32)
x2 = torch.tensor(x2, requires_grad=False, dtype=torch.float32)

T = 20
p_w = 10**(a[0, 0] - a[0, 1]/(T + a[0, 2])) #
p_d = 10**(a[1, 0] - a[1, 1]/(T + a[1, 2]))
p_sol = np.array([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])
p_sol = torch.tensor(p_sol, requires_grad=False, dtype=torch.float32)

A = Variable(torch.tensor([1.0, 1.0]), requires_grad=True)

alpha = 0.0001
eps = 0.001

p_est = Calc_P(x1, x2, p_w, p_d, A)
loss = (p_est - p_sol)**2
loss = loss.sum()
loss.backward()
A_GRAD = float(torch.norm(A.grad))
iter = 0

while A_GRAD >= eps:
    p_est = Calc_P(x1, x2, p_w, p_d, A)
    loss = (p_est - p_sol)**2
    loss = loss.sum()
    loss.backward()

    with torch.no_grad():
        A -= alpha*A.grad
        A_GRAD = float(torch.norm(A.grad))
        A.grad.zero_()

    iter += 1
print('estimation of A_12 & A_21 is:', A)
print('final loss is:', loss.data.numpy())
print('total iterations:', iter)

Plot_Results(p_sol, Calc_P(x1, x2, p_w, p_d, A), x1)