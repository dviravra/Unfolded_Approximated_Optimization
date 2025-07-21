import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import time
from datetime import datetime

# ================ Preparations ====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
datatype = torch.float32

# ================ Parameters =======================
r = 5  #  rank
d1 = 1000  # number of rows
d2 = 1000  # number of columns
alpha = 0.1  # fraction of outliers
step_initial = 1.5  # initial step size (matrix version)
ths_initial = 1e-3  # initial threshold value 
maxIt = 17  # number of layers to train
loss_list = []

# ============= Generate RPCA problems ==============
def generate_problem(r, d1, d2, alpha):
    U0_t = torch.randn(d1, r, dtype=datatype, device=device) / math.sqrt(d1)
    V0_t = torch.randn(d1, r, dtype=datatype, device=device) / math.sqrt(d2)
    idx = torch.randperm(d1 * d2, device=device)
    idx = idx[:math.floor(alpha * d1 * d2)]
    Y0_t = torch.mm(U0_t, V0_t.t())
    Y0_t = Y0_t.reshape(-1)
    s_range = torch.mean(torch.abs(Y0_t))
    S0_t = torch.rand(len(idx), dtype=datatype, device=device)
    S0_t = s_range * (2.0 * S0_t - 1.0)
    Y0_t[idx] = Y0_t[idx] + S0_t
    Y0_t = Y0_t.reshape((d1, d2))
    return U0_t, V0_t, Y0_t

# Save a new data file
U0_t, V0_t, Y0_t = generate_problem(r, d1, d2, alpha)
X_star = torch.mm(U0_t, V0_t.t())
U0_np = U0_t.cpu().numpy()
V0_np = V0_t.cpu().numpy()
Y0_np = Y0_t.cpu().numpy()
X_star_np = X_star.cpu().numpy()

now = datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H-%M")
filename = f"n{d1}_r{r}_alpha{alpha}_eta_is_matrix_{timestamp}.mat"
os.makedirs("data", exist_ok=True)
filepath = os.path.join("data", filename)
sio.savemat(filepath, {
    "U_star": U0_np,
    "V_star": V0_np,
    "Y_star": Y0_np,
    "X_star": X_star_np
})
print(f"Saved data file to {filepath}")

# Create approximate
approx_iters_u = [4,6,8,10,12,14]
approx_iters_v = [3,5,7,9,11,13,15]
full_update_mask_U = []
full_update_mask_V = []
for t in range(1, maxIt):
    full_update_mask_U.append(False if t in approx_iters_u else True)
    full_update_mask_V.append(False if t in approx_iters_v else True)


# =================== LRPCA Model ====================
class MatNet(nn.Module):
    def __init__(self):
        super(MatNet, self).__init__()
        self.ths_v = [nn.Parameter(torch.tensor(ths_initial, dtype=datatype, device=device), requires_grad=True)
                      for _ in range(maxIt)]
        self.step_U = [
            nn.Parameter(torch.full((d1, r), step_initial, dtype=datatype, device=device), requires_grad=True)
            for _ in range(maxIt)]
        self.step_V = [
            nn.Parameter(torch.full((d2, r), step_initial, dtype=datatype, device=device), requires_grad=True)
            for _ in range(maxIt)]
        self.ths_backup = [torch.tensor(0.0, dtype=datatype, device=device) for _ in range(maxIt)]

    def thre(self, inputs, threshold):
        return torch.sign(inputs) * torch.max(torch.abs(inputs) - threshold, torch.zeros_like(inputs))

    def forward(self, Y0_t, r, U0_t, V0_t, num_l, full_update_mask_U, full_update_mask_V):
        # Initialization 
        S_t = self.thre(Y0_t, self.ths_v[0])
        L, Sigma, R_mat = torch.svd_lowrank(Y0_t - S_t, q=r, niter=4)
        Sigsqrt = torch.diag(torch.sqrt(Sigma))
        U_t = torch.mm(L, Sigsqrt)
        V_t = torch.mm(R_mat, Sigsqrt)
        # Main iterative 
        for t in range(1, num_l):
            YmUV = Y0_t - torch.mm(U_t, V_t.t())
            S_t = self.thre(YmUV, self.ths_v[t])
            E_t = YmUV - S_t
            # ----- Update for U -----
            if full_update_mask_U[t - 1]:
                # Full update
                Vkernel = torch.inverse(torch.mm(V_t.t(), V_t))
                Unew = U_t + self.step_U[t] * (torch.mm(E_t, V_t) @ Vkernel)
            else:
                # Approximate update
                Unew = U_t
            U_t = Unew
            # ----- Update for V -----
            if full_update_mask_V[t - 1]:
                Ukernel = torch.inverse(torch.mm(U_t.t(), U_t))
                Vnew = V_t + self.step_V[t] * (torch.mm(U_t.t(), E_t).t() @ Ukernel)
            else:
                Vnew = V_t
            V_t = Vnew
        L_approx = torch.mm(U_t, V_t.t())
        S_approx = Y0_t - L_approx
        loss = (L_approx - torch.mm(U0_t, V0_t.t())).norm()
        return loss, L_approx, S_approx

    def InitializeThs(self, en_l):
        self.ths_v[en_l].data = self.ths_v[en_l - 1].data * 0.1

    def CheckNegative(self):
        isNegative = any(th.data < 0 for th in self.ths_v)
        if isNegative:
            for t in range(maxIt):
                self.ths_v[t].data = self.ths_backup[t].clone()
        else:
            for t in range(maxIt):
                self.ths_backup[t] = self.ths_v[t].data.clone()
        return isNegative

    def EnableSingleLayer(self, en_l):
        for t in range(maxIt):
            self.ths_v[t].requires_grad = False
            self.step_U[t].requires_grad = False
            self.step_V[t].requires_grad = False
        self.ths_v[en_l].requires_grad = True
        self.step_U[en_l].requires_grad = True
        self.step_V[en_l].requires_grad = True

    def EnableLayers(self, num_l):
        for t in range(num_l):
            self.ths_v[t].requires_grad = True
            self.step_U[t].requires_grad = True
            self.step_V[t].requires_grad = True
        for t in range(num_l, maxIt):
            self.ths_v[t].requires_grad = False
            self.step_U[t].requires_grad = False
            self.step_V[t].requires_grad = False


# ================= Training Scripts =======================
Nepoches_pre = 500
Nepoches_full = 1000
lr_fac = 1.0

net = MatNet()
optimizers = []
for i in range(maxIt):
    optimizer = optim.SGD({net.ths_v[i]}, lr=lr_fac * ths_initial / 5000.0)
    optimizer.add_param_group({'params': [net.step_U[i]], 'lr': lr_fac * 0.1})
    optimizer.add_param_group({'params': [net.step_V[i]], 'lr': lr_fac * 0.1})
    optimizers.append(optimizer)

start = time.time()
for stage in range(maxIt):
    print('Layer', stage, ', Pre-training ======================')
    if stage > 6:
        Nepoches_full = 500
    if stage > 0:
        optimizers[stage].param_groups[0]['lr'] = net.ths_v[stage - 1].data * lr_fac / 5000.0
    for epoch in range(Nepoches_pre):
        for i in range(maxIt):
            optimizers[i].zero_grad()
        U0_t, V0_t, Y0_t = generate_problem(r, d1, d2, alpha)
        net.EnableSingleLayer(stage)
        if stage > 0:
            net.InitializeThs(stage)
        loss, L_approx, S_approx = net(Y0_t, r, U0_t, V0_t, stage + 1, full_update_mask_U, full_update_mask_V)
        loss.backward()
        optimizers[stage].step()
        loss_list.append(loss.item())
        if epoch % 10 == 0:
            if net.CheckNegative():
                print("Negative detected, restored")
        if epoch % 20 == 0:
            print("epoch:", epoch, "\t loss:", loss.item())
    print('Layer', stage, ', Full-training ======================')
    if stage == 0:
        continue
    for epoch in range(Nepoches_full):
        for i in range(maxIt):
            optimizers[i].zero_grad()
        U0_t, V0_t, Y0_t = generate_problem(r, d1, d2, alpha)
        net.EnableLayers(stage + 1)
        loss, L_approx, S_approx = net(Y0_t, r, U0_t, V0_t, stage + 1, full_update_mask_U, full_update_mask_V)
        loss.backward()
        for i in range(stage + 1):
            optimizers[i].step()
        loss_list.append(loss.item())
        if epoch % 20 == 0:
            print("epoch:", epoch, "\t loss:", loss.item())

end = time.time()
print("Training end. Time:", end - start)

# ===================== Save model to .mat file ========================
result_ths = np.zeros((maxIt,))
result_stepU = []
result_stepV = []
for i in range(maxIt):
    result_ths[i] = net.ths_v[i].data.cpu().numpy()
    result_stepU.append(net.step_U[i].data.cpu().numpy())
    result_stepV.append(net.step_V[i].data.cpu().numpy())
loss_array = np.array(loss_list)
spath = f"LRPCA_alpha_{alpha}_{timestamp}.mat"
sio.savemat(spath, {
    'ths': result_ths,
    'stepU': np.stack(result_stepU, axis=0),
    'stepV': np.stack(result_stepV, axis=0),
    'loss_list': loss_array
})
