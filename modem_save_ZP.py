import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import scipy.io
import mat73
import scipy.io
# Basic parameters
M = 256 + 64 # length of time domain samples
N = 256 # The number of subcarriers
rho = 0.01 # 1/SNR (20 dB)
lambda_mse = 0.9 # weight for MSE between 2 modem/demodem matrices
D = 2 # The bandwidth reserved
# transformer parameters
seq_len = N 
feature_dim = 2
d_model_trans_modu = M * 2
d_model_trans_demodu = N * 2

# Read datasets from .mat files.
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

test_dataset = GenDataset(H_test)

# Create data loaders.
batch_size_test = 50
test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

for (H1,H2) in test_dataloader:
    print(f"Shape of H [N, C, H, W]: {H1.shape}")
    # print(f"Shape of y: {y.shape} {y.dtype}")
    break

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = torch.device("cuda:1" if torch.cuda.is_available else "cpu")
print(f"Using {device} device") 

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
model = torch.load('model_bestloss.pth')
print(f"modulation optimization network:\n")
print(model)



def save_results(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    mtx_mod = torch.zeros(2,N,N,device=device)
    with torch.no_grad():
        for (H1,H2) in dataloader:
            H1, H2 = H1.to(device), H2.to(device)
            mtx_mod1 = model(H1) 
            mtx_mod2 = model(H2)  
            mtx_mod_nowbatch = (mtx_mod1 + mtx_mod2) / 2
            mtx_mod_nowbatch = mtx_mod_nowbatch.mean(dim=0)
            mtx_mod += mtx_mod_nowbatch 
    mtx_mod /=  num_batches
    mtx_mod_c = torch.complex(mtx_mod[0,:,:],mtx_mod[1,:,:])
    mtx_mod_c,_ = torch.linalg.qr(mtx_mod_c,mode='reduced')
    mtx_mod[0,:,:] = mtx_mod_c.real
    mtx_mod[1,:,:] = mtx_mod_c.imag
    output_data1 = {
        'F': mtx_mod.cpu(),
    }
    scipy.io.savemat("F.mat", output_data1)
    print(f"The modulation matrix have been saved to F.mat\n")

save_results(test_dataloader,model)