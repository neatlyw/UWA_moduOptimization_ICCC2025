import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import scipy.io
import mat73
import scipy.io
import numpy as np
# Basic parameters
M = 256 + 64 # length of time domain samples
N = 256 # The number of subcarriers
rho = 0.01 # 1/SNR (30 dB)
eta = 0.005
lambda_mse = 1 - eta # weight for MSE between 2 modem/demodem matrices
D = 2 # The bandwidth reserved
# transformer parameters
seq_len = N 
feature_dim = 2
'''
d_model_trans_modu = M * 2
d_model_trans_demodu = N * 2
'''

# Read datasets from .mat files.
H_train = mat73.loadmat('H_train.mat') # dimension:N*M
H_train = H_train['H_train']
H_test = mat73.loadmat('H_test.mat')
H_test = H_test['H_test']

# deal with the dataset
class GenDataset(Dataset):
    def __init__(self, H_data):
        self.H_data = H_data

    def __len__(self):
        return len(self.H_data) // 2

    def __getitem__(self, idx):
        # get data
        H1 = self.H_data[2*idx]
        H2 = self.H_data[2*idx+1]
        return torch.tensor(H1, dtype=torch.float32), torch.tensor(H2, dtype=torch.float32)

train_dataset = GenDataset(H_train)
test_dataset = GenDataset(H_test)

# Create data loaders.
batch_size_train = 200
batch_size_test = 50
num_batch_progress = 4000 // (2*batch_size_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

for (H1,H2) in test_dataloader:
    print(f"Shape of H [N, C, H, W]: {H1.shape}")
    # print(f"Shape of y: {y.shape} {y.dtype}")
    break

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device") 

'''
# Define Residual block
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=4, kernel_size=7, padding=3),
            nn.BatchNorm2d(4),
            nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=2, kernel_size=7, padding=3),
            nn.BatchNorm2d(2),
            nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=2, kernel_size=7, padding=3),
            nn.BatchNorm2d(2),
            nn.ReLU())

    def forward(self, x):
        output_1 = self.conv1(x)
        output_1 = torch.cat((output_1,x), dim=1)
        output_2 = self.conv2(output_1)
        output_2 = torch.cat((output_2, output_1), dim=1)
        output_final = self.conv3(output_2)
        return output_final
'''
# Define model
class moduNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.myNetwork = nn.Sequential(
            #nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model_trans_modu, nhead=16, 
                #dropout=0.2, activation="gelu", batch_first=True), num_layers=8
            #),
            nn.Flatten(),
            nn.Linear(2*M*N,8*N),
            nn.GELU(),
            nn.Linear(8*N,4*N),
            nn.GELU(),
            nn.Linear(4*N,4*N),
            nn.GELU(),
            nn.Linear(4*N,2*N*N)
        )
    def forward(self,H):
        # H_reshape = torch.permute(H,(0,2,1,3))
        # H_reshape = H_reshape.reshape(H.size(0),N,2*M)
        # modulation matrix
        output = self.myNetwork(H)
        # output = output[:,:,:2*N]
        mtx_mod1 = output.reshape(output.size(0),2,N, N)
        #mtx_mod1 = mtx_mod1.permute(0,2,1,3)
        mtx_mod = torch.zeros_like(mtx_mod1)
        mtx_mod1_c = torch.complex(mtx_mod1[:,0,:,:],mtx_mod1[:,1,:,:])
        mtx_mod_c,_ = torch.linalg.qr(mtx_mod1_c) # orthogonalization and normalize
        mtx_mod[:,0,:,:] = mtx_mod_c.real
        mtx_mod[:,1,:,:] = mtx_mod_c.imag
        return mtx_mod
    

model = moduNetwork().to(device)
print(f"modem optimization network:\n")
print(model)

# define the loss function
class Loss_my(nn.Module):
    def __init__(self):
        super(Loss_my, self).__init__()

    def forward(self, H1, mtx_mod1, H2, mtx_mod2):
        H1_c = torch.complex(H1[:,0,:,:],H1[:,1,:,:])
        mtx_mod1_c = torch.complex(mtx_mod1[:,0,:,:],mtx_mod1[:,1,:,:])
        H2_c = torch.complex(H2[:,0,:,:],H2[:,1,:,:])
        mtx_mod2_c = torch.complex(mtx_mod2[:,0,:,:],mtx_mod2[:,1,:,:])
        # equivalent channels and power
        H1e_c = torch.matmul(H1_c,mtx_mod1_c)
        H2e_c = torch.matmul(H2_c,mtx_mod2_c)
        # compute the covariance mtx
        identity_matrices = torch.eye(N, dtype=torch.complex64).unsqueeze(0).repeat(H1_c.size(0), 1, 1).to(device)
        # identity_matrices = identity_matrices.to(device)
        cov1 = torch.linalg.inv(torch.matmul(torch.transpose(torch.conj(H1e_c),1,2),H1e_c)+rho*identity_matrices)
        cov2 = torch.linalg.inv(torch.matmul(torch.transpose(torch.conj(H2e_c),1,2),H2e_c)+rho*identity_matrices)
        # compute total mse
        mse1 = torch.diagonal(cov1.real,dim1=1,dim2=2)
        mse1_sum = torch.sum(mse1,dim=1)
        mse1 = mse1 / mse1_sum.unsqueeze(1)
        mse2 = torch.diagonal(cov2.real,dim1=1,dim2=2)
        mse2_sum = torch.sum(mse2,dim=1)
        mse2 = mse2 / mse2_sum.unsqueeze(1)
        mse_best = torch.full_like(mse1, 1.0 / N)
        distance1 = F.mse_loss(mse1,mse_best)
        distance1 = distance1 * N * N
        distance2 = F.mse_loss(mse2,mse_best)
        distance2 = distance2 * N * N
        nmse_data = ( distance1 + distance2 ) / 2 
        # nmse_data_log = torch.log(1+nmse_data)
        # nmse_data = nmse_data.mean()
        # mse between mod/demod matrices
        nmse_mod = F.mse_loss(mtx_mod1, mtx_mod2)
        nmse_mod = nmse_mod * 2 * N 
        # nmse_mod_log = torch.log(1+nmse_mod)
        # compute MSE
        # loss = (1-lambda_mse)*nmse_data_log + lambda_mse*nmse_mod_log
        loss = (1-lambda_mse)*nmse_data + lambda_mse*nmse_mod
        return nmse_data, nmse_mod, loss

loss_fn = Loss_my()
optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    train_nmse_data = 0
    train_nmse_mod = 0
    for batch, (H1, H2) in enumerate(dataloader):
        H1, H2 = H1.to(device), H2.to(device)
        mtx_mod1 = model(H1) 
        mtx_mod2 = model(H2)  
        nmse_data, nmse_mod, loss = loss_fn(H1, mtx_mod1, H2, mtx_mod2)
        train_loss += loss.item()
        train_nmse_data += nmse_data.item()
        train_nmse_mod += nmse_mod.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (batch+1) % num_batch_progress == 0:
            loss, current = loss.item(), (batch + 1) * len(H1)
            nmse_data = nmse_data.item()
            nmse_mod = nmse_mod.item()
            print(f"loss: {loss:>10.4e} Data NMSE: {nmse_data:>10.4e} Modulation NMSE:{nmse_mod:10.4e} [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    train_nmse_mod /= num_batches
    train_nmse_data /= num_batches
    return train_nmse_data,train_nmse_mod,train_loss

def test(dataloader, model, loss_fn, loss_best, epoch_best, epoch_num):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    test_nmse_mod = 0
    test_nmse_data = 0
    with torch.no_grad():
        for (H1,H2) in dataloader:
            H1, H2 = H1.to(device), H2.to(device)
            mtx_mod1 = model(H1) 
            mtx_mod2 = model(H2)  
            nmse_data, nmse_mod, loss = loss_fn(H1, mtx_mod1, H2, mtx_mod2)
            test_loss += loss.item()
            test_nmse_mod += nmse_mod.item()
            test_nmse_data += nmse_data.item()
    test_loss /= num_batches
    test_nmse_mod /= num_batches
    test_nmse_data /= num_batches
    print(f"loss: {test_loss:>10.4e} Data NMSE: {test_nmse_data:>10.4e} Modulation NMSE:{test_nmse_mod:10.4e}")
    if test_loss<loss_best:
        torch.save(model, 'model_bestloss.pth')
        loss_best = test_loss
        epoch_best = epoch_num
        print(f"The best model has been saved with loss: {test_loss:>10.4e} Data NMSE: {test_nmse_data:>10.4e} Modem NMSE:{test_nmse_mod:10.4e} for the {epoch_num+1}-th epoch")
    return loss_best,epoch_best,test_nmse_data,test_nmse_mod,test_loss

epochs = 2500
loss_best = 10
epoch_best = 0
train_loss_list=[]
train_nmse_data_list=[]
train_nmse_mod_list=[]
test_loss_list=[]
test_nmse_data_list=[]
test_nmse_mod_list=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_nmse_data,train_nmse_mod,train_loss=train(train_dataloader, model, loss_fn, optimizer)
    train_loss_list.append(train_loss)
    train_nmse_data_list.append(train_nmse_data)
    train_nmse_mod_list.append(train_nmse_mod)
    train_loss_arr = np.array(train_loss_list)
    train_nmse_data_arr = np.array(train_nmse_data_list)
    train_nmse_mod_arr = np.array(train_nmse_mod_list)
    scipy.io.savemat("train_loss_epoch.mat", {"train_loss_epoch": train_loss_arr})
    scipy.io.savemat("train_nmse_data_epoch.mat", {"train_nmse_data_epoch": train_nmse_data_arr})
    scipy.io.savemat("train_nmse_mod_epoch.mat", {"train_nmse_mod_epoch": train_nmse_mod_arr})
    loss_best,epoch_best,test_nmse_data,test_nmse_mod,test_loss=test(test_dataloader, model, loss_fn, loss_best, epoch_best,t)
    test_loss_list.append(test_loss)
    test_nmse_data_list.append(test_nmse_data)
    test_nmse_mod_list.append(test_nmse_mod)
    test_loss_arr = np.array(test_loss_list)
    test_nmse_data_arr = np.array(test_nmse_data_list)
    test_nmse_mod_arr = np.array(test_nmse_mod_list)
    scipy.io.savemat("test_loss_epoch.mat", {"test_loss_epoch": test_loss_arr})
    scipy.io.savemat("test_nmse_data_epoch.mat", {"test_nmse_data_epoch": test_nmse_data_arr})
    scipy.io.savemat("test_nmse_mod_epoch.mat", {"test_nmse_mod_epoch": test_nmse_mod_arr})
    print(f"The best loss occurs at epoch {epoch_best+1}\n")
print("Done!")
