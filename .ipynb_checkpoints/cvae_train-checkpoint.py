import time
import h5py
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adni_util import get_normalized_data

CDR_MAX_SCORE = 18



def create_cohort_hdf5(hdf5_path, nii_dir, normalize=False):
    '''
    hdf5_path: path to the new HDF5 file to consolidate data into
    nii_dir: directory containing the data as nii files.
    normalize: whether to normalize the images or not before writing
    '''
    imgs, img_ids, patient_info = get_normalized_data(nii_dir, normalize=normalize)

    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('imgs', data=imgs)
        f.create_dataset('img_ids', data=np.array(img_ids, dtype=int))
        f.create_dataset('patient_info', data=np.array(patient_info, dtype='S'))
        

class PETDataset(Dataset):
    
    def __init__(self, hdf5_path, all_cohorts_path, diagnosis=3, normalize='scale'):
        '''
        
        '''
        # Read data from files
        self.hdf5_path = hdf5_path
        self.all_cohorts_df = pd.read_csv(all_cohorts_path).set_index('image_id')
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        # Get image data
        self.imgs = self.hdf5_file['imgs'][:].astype(np.float32)
        # Filter the image IDs: HDF5 file does not account for missing EXAMDATE, VISDATE, or CDRSB; but cohorts DataFrame does
        # Image IDs that have not been filtered:
        self.all_img_ids = pd.Index(np.array(self.hdf5_file.get('img_ids')).astype(int))
        # Image IDs that have been filtered:
        self.img_ids = self.all_img_ids.intersection(self.all_cohorts_df[self.all_cohorts_df['DIAGNOSIS'] == diagnosis].index)
        self.imgs = self.imgs[self.all_img_ids.get_indexer(self.img_ids)]
        # Crop images to (90 x 108 x 90)
        self.imgs = self.imgs[:, 1:-1, 1:-1, 1:-1]
        
        if normalize == 'scale':
            imgs_max, imgs_min = self.imgs.max(axis=(1,2,3), keepdims=True), self.imgs.min(axis=(1,2,3), keepdims=True)
            self.imgs = (self.imgs - imgs_min) / (imgs_max - imgs_min)
        elif normalize == 'sigmoid':
            self.imgs = 1 / (1 + np.exp(-self.imgs))
        elif normalize == 'clip':
            self.imgs = np.clip(self.imgs, 0., 1.)
        
        self.cdr_scores = (self.all_cohorts_df.loc[self.img_ids]['CDRSB'] / CDR_MAX_SCORE).to_numpy().astype(np.float32)

    
    
    def __len__(self):
        return self.imgs.shape[0]
    
    
    def __getitem__(self, idx):
        img = torch.unsqueeze(torch.Tensor(self.imgs[idx].copy()), 0)
        cdr_score = self.cdr_scores[idx].copy()
        return img, cdr_score
    
    
class PETsMRIDataset(Dataset):
    
    def __init__(self, pet_hdf5_path, mri_hdf5_path, all_cohorts_path, diagnosis=3, noramalize='scale'):
        '''
        
        '''
        # Read data from files
        self.pet_hdf5_path = pet_hdf5_path
        self.mri_hdf5_path = mri_hdf5_path
        self.pet_hdf5_file = h5py.File(self.pet_hdf5_path, 'r')
        self.mri_hdf5_file = h5py.File(self.mri_hdf5_path, 'r')
        
        # Unique PET images are matched to their closest sMRI image; therefore the sMRI image ID column is not unique
        self.all_cohorts_df = pd.read_csv(all_cohorts_path).set_index('image_id_pet')         
        # Get image data
        self.pet_imgs = self.pet_hdf5_file['imgs'][:].astype(np.float32)
        self.mri_imgs = self.mri_hdf5_file['imgs'][:].astype(np.float32)
        
        # Filter the image IDs: HDF5 file does not account for missing EXAMDATE, VISDATE, or CDRSB; but cohorts DataFrame does
        # Image IDs that have not been filtered:
        self.all_pet_img_ids = pd.Index(np.array(self.pet_hdf5_file.get('img_ids')).astype(int))
        self.all_mri_img_ids = pd.Index(np.array(self.mri_hdf5_file.get('img_ids')).astype(int))


class cVAE_MNIST(nn.Module):
    def __init__(self, feat_dim, latent_dim, num_classes):
        super().__init__()
        
        # Encoder layers
        # Provide the label as the condition
#         self.flatten = nn.Flatten()
        hidden_dim = 400
        self.linear1 = nn.Linear(feat_dim+num_classes, hidden_dim)
        self.linear2_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear2_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.linear3 = nn.Linear(latent_dim+num_classes, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, feat_dim)
        
        self.sigmoid = nn.Sigmoid()
        
        
    def encode(self, x, y):
        '''
        x: data input (N x feat_dim)
        y: condition (N x num_classes)
        '''
        xy = torch.cat([x, y], dim=1)
        out1 = self.linear1(xy)
        out2 = out1.relu()
        # Predict distribution parameters
        z_mu = self.linear2_mu(out2)
        z_var = self.linear2_var(out2)
        return z_mu, z_var
    
    
    def decode(self, z, y):
        '''
        z: latent features (N x latent_size)
        y: condition (N x num_classes)
        '''
        zy = torch.cat([z, y], dim=1)
        out1 = self.linear3(zy)
        out2 = out1.relu()
        out3 = self.linear4(out2)
        return self.sigmoid(out3)
    
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        epsilon = torch.randn_like(std)
        return mu + epsilon*std
        
        
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y) 
        return x_hat, mu, logvar
    
    
    
class cVAE_PET(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
#         self.relu = nn.ReLU()
        self.latent_dim = latent_dim
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, (5,5,5), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv3d(16, 16, (5,5,5), stride=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 64, (5,5,5), stride=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3,3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, 32),
            nn.ReLU()
        )
        
        self.linear_mu = nn.Linear(33, latent_dim)
        self.linear_var = nn.Linear(33, latent_dim)
        
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 2304),
            nn.ReLU()  
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (3,3,3)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 16, (5,5,5), stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 16, (5,5,5), stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, (5,5,5), stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )
        

        

#         hidden_dim = 400
#         self.linear1 = nn.Linear(feat_dim+num_classes, hidden_dim)
#         self.linear2_mu = nn.Linear(hidden_dim, latent_dim)
#         self.linear2_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
#         self.linear3 = nn.Linear(latent_dim+num_classes, hidden_dim)
#         self.linear4 = nn.Linear(hidden_dim, feat_dim)
        
#         self.sigmoid = nn.Sigmoid()
        
        
    def encode(self, x, y):
        '''
        x: data input (N x feat_dim)
        y: condition (N x num_classes)
        '''
        
        out1 = self.encoder(x)
        out2 = torch.cat([out1, y.unsqueeze(1)], dim=1)
        
        z_mu = self.linear_mu(out2)
        z_var = self.linear_var(out2)
        return z_mu, z_var
    
    
    def decode(self, z, y):
        '''
        z: latent features (N x latent_size)
        y: condition (N x num_classes)
        '''
        zy = torch.cat([z, y.unsqueeze(1)], dim=1)
        out1 = self.decoder1(zy)
        out2 = out1.reshape(-1, 64, 3, 4, 3)
        out3 = self.decoder2(out2)
        return out3
    
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        epsilon = torch.randn_like(std)
        return mu + epsilon*std
        
        
    def forward(self, x, y):
#         return self.encoder(x)
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y) 
        return x_hat, mu, logvar
    
    
    
    
def train_model(model, optimizer, dataset, loss_fn, epochs, batch_size, save_freq=None, save_path=None, scheduler=None, device='cpu'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    print(f'Scheduler: {scheduler}' if scheduler else 'No learning rate scheduling!')
    print(f'Training for {epochs} epochs, with batch size={batch_size}')
    print(f'Using device: {device}')
    print(f'Saving model every {save_freq} epochs to {save_path}' if save_freq else 'WARNING: Will not save model!')

    for e in range(epochs):
        losses = []
        all_pred, all_true = [], []
        t = time.time()
        print(f'\n-----Epoch {e+1}/{epochs}-----')
        for i, (x, labels) in enumerate(loader):
#             labels = one_hot(labels, 10).to(device)
            labels = labels.to(device)
            x = x.to(device)#.squeeze().flatten(start_dim=1)
            pred, mu, logvar = model(x, labels)
            loss = loss_fn(x, pred, mu, logvar)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            all_pred.append(pred.cpu())
            all_true.append(labels.cpu())

            if len(losses) == 10 or i == len(loader)-1:
                elapsed = time.time() - t
                t = time.time()
                print(f'Batch {i+1}/{len(loader)}, loss: {np.mean(losses)} ({elapsed:.3f}s)')
                losses = []
                
        if scheduler is not None:
            scheduler.step()
            
        if save_freq and ((e+1) % save_freq == 0 or e == epochs-1):
            save_model(save_path, model, optimizer, epochs)
            print(f'Saved to {save_path}')       
         
#         f1 = f1_score(torch.cat(all_true, dim=0), 
#                       (torch.sigmoid(torch.cat(all_pred, dim=0)) > 0.5).type(torch.float), 
#                       average='weighted')
#         print(f'F1 score: {f1}')


def cvae_loss_fn(x, x_hat, mu, logvar):
    bce_loss = F.binary_cross_entropy(x_hat, x, reduction='none')
    bce_loss = torch.mean(torch.sum(bce_loss, dim=(1,2,3,4)))
#     bce_loss = F.mse_loss(x_hat, x)
#     l1 = F.binary_cross_entropy(x_hat, x, reduction='none')
#     print(torch.sum(l1, dim=(1,2,3,4)))
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    print(f'kld: {kld_loss}, bce: {bce_loss}')
    return bce_loss + kld_loss


def save_model(save_path, model, optimizer, epoch):
    '''Save a model to disk'''
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
    
    
def load_model(model, save_path, strict=True):
    '''Load a previously saved model for inference'''
    checkpoint = torch.load(save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    return model
