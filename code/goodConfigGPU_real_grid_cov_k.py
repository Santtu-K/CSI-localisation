import torch
import pdb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm
#######################################################################
##########-------------------- Data Loader ------------------##########
grid = 3
k = 32
offline_data = h5py.File('../pythonTrain_grid' + str(grid) + '_r_real_cov_k' + str(k) + '.mat', 'r')

key_offline = list(offline_data.keys())
labels_offline = np.array(offline_data.get(key_offline[1]))  # physical pairwise distances
feature_offline = np.array(offline_data.get(key_offline[0]))  # CSI features

print('Train/offline data:', key_offline)
print('Training/offline label shape:', labels_offline.shape)
print('Training/offline raw feature shape:', feature_offline.shape)

# Input data size:    grid^2 by 32 by 2
# Label size:      grid^2 by 1

# GPU stuff
if torch.cuda.is_available():
    dev = "cuda:0"
    print("GPU available")
else: 
    dev = "cpu"
    print("GPU NOT available")
device = torch.device(dev)

dataset_size = feature_offline.shape[0]
# shuffle inputs
indsh = (np.arange(dataset_size))
np.random.shuffle(indsh)
phy_dist = labels_offline.copy()
phy_dist = torch.Tensor(phy_dist[indsh, :])
print(phy_dist.get_device())
phy_dist = phy_dist.to(device) # Convert to GPU
print(phy_dist.get_device())

f = feature_offline.copy()
feat_mean = np.mean(f)
feat_std = np.std(f)
# normalize feature
# x = f - feat_mean
# x = x / feat_std
x = torch.Tensor(f[indsh, :, :])
x = x.to(device) # Convert to GPU


ratio = 0.5
X_train = x[:int(ratio * dataset_size), :]
y_train = phy_dist[:int(ratio * dataset_size), :]
# generate validation data
X_val = x[int(ratio * dataset_size):, :]
y_val = phy_dist[int(ratio * dataset_size):, :]

#######################################################################
##########------------------ Network Structure --------------##########


n_feat = 32
layer_configs = [[512,256,128,64,32,16]] #[256,128,32,16,8,4],

for layers in layer_configs:
    n_layer = len(layers)

    filepath = "../bestLayers/"
    filename = "n_" + str(grid) + "_" + str(n_layer) + "layer_real.txt"
    f = open(filepath + filename, "a")
    f.write("Used layers: ")
    print("Current layers: ")
    for layer in layers:
        f.write(str(layer) + ", ")
        print(str(layer) + ", ")
    # Define the shared layers to be used in each block
    shared_layers = nn.Sequential(
        nn.Linear(n_feat, layers[0], bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(layers[0]),
        nn.Linear(layers[0], layers[1], bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(layers[1]),
        nn.Linear(layers[1], layers[2], bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(layers[2]),
        nn.Linear(layers[2], layers[3], bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(layers[3]),
        nn.Linear(layers[3], layers[4], bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(layers[4]),
        nn.Linear(layers[4], layers[5], bias=True)
    )


    class My_NN(nn.Module):
        def __init__(self):
            super(My_NN, self).__init__()
            self.block = shared_layers

        def _initialize_weights(self):
            # print('initialized parameters')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, feature):  # input feature matrix size: batch_size * 32 * 2
            output1 = self.block(feature[:, :, 0])
            output2 = self.block(feature[:, :, 1])

            return torch.norm(output1 - output2, dim=1).unsqueeze(1)


    # Initialize the model and other hyperparameters
    my_nn = My_NN()
    my_nn = my_nn.to(device)
    criterion = nn.MSELoss(reduction='sum')

    def custom_loss(my_outputs, my_labels):
        #pdb.set_trace()
        sub = torch.sub(my_outputs, my_labels, alpha=1)
        sqr = torch.square(sub)
        eps= torch.add(my_labels,0.0001, alpha=1)
        div = torch.div(sqr,eps)
        t_sum = torch.sum(div, dtype= torch.double)
        # print(sub)
        # print(sqr)
        # print(div)
        # print(t_sum)
        return t_sum

    
    optimizer = torch.optim.Adam(list(my_nn.parameters()), lr=1e-2)
    # Define a learning rate scheduler based on validation loss
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5, threshold=1e-5, verbose=True)
    #
    # Training loop
    my_nn.train()
    num_epochs = 1000
    print_parameters_every = 20  # Print parameters every n epochs
    train_losses = []
    validation_losses = []

    for epoch in tqdm(range(num_epochs)):
        output = my_nn(X_train)  # input is 2 features; output is (z1-z2)
        loss = criterion(output, y_train)#custom_loss(output, y_train)  # calculate MSE loss of (z1-z2) and physical distance
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation loop
        with torch.no_grad():
            output = my_nn(X_val)
            val_loss = criterion(output, y_val)
            validation_losses.append(val_loss.item())
            #scheduler.step(val_loss)
    # Plot the training and validation loss
    plt.figure(1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("RMSE_real.pdf")

    # Testing/online data
    online_data = h5py.File('../pythonTest_grid' + str(grid) + '_r_real_cov.mat', 'r')
    key_online = list(online_data.keys())
    print('Test/online data:', key_online)

    labels_online = np.array(online_data.get(key_online[1]))  # physical pairwise distances
    feature_online = np.array(online_data.get(key_online[0]))  # CSI features

    print('Testing/online label shape:', labels_online.shape)
    print('Testing/online raw feature shape:', feature_online.shape)

    y_test_raw = labels_online.copy()
    x_test_raw = feature_online.copy()

    y_test = torch.Tensor(y_test_raw).to(device)
    X_test = torch.Tensor(x_test_raw).to(device)

    # test loop
    my_nn.eval()
    errors_RMSE = []
    feature_d = []
    physical_d = []
    with torch.no_grad():
        pred = my_nn(X_test)
        error_rmse = ((pred.cpu() - y_test.cpu()) ** 2).numpy()
        errors_RMSE.extend(error_rmse)
        feature_d.extend(pred.detach().cpu().numpy())
        physical_d.extend(y_test.detach().cpu().numpy())
    RMSE = np.sqrt(np.sum(errors_RMSE) / len(errors_RMSE))
    print('Test RMSE is', RMSE)
    f.write("Test RMSE is: " +str(RMSE) + "\n")
    f.close()
    plt.figure(2)
    feature_d_norm, physical_d_norm = np.array(feature_d), np.array(physical_d)
    feature_d_norm/=np.max(feature_d_norm)
    physical_d_norm/=np.max(physical_d_norm)
    plt.plot(physical_d_norm,feature_d_norm, '*')
    plt.ylabel('predicted distance')
    plt.xlabel('true distance')
    plt.title('Physical and feature distances with a Metric Learning step')
    plt.savefig("Phys_vs_Feat_real.pdf")

    filepath_data = "../outputs/"
    filename_data =  "grid" + str(grid) + "_" + str(n_layer) + "layer_" + '_'.join(str(x) for x in layers)
    scipy.io.savemat(filepath_data + filename_data + '_r_real_cov_k' + str(k) + '.mat', {'z_d': np.array(feature_d), 'phys_d': np.array(physical_d)})




