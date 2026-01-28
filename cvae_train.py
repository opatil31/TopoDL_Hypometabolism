import time
import h5py
import torch
import os
import re
import nibabel as nib
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from adni_util import scale_normalize

CDR_MAX_SCORE = 18

def create_hdf5_dataset(imgs_path, out_file_path, img_id_pattern):
    '''images should all be of the same shape'''
    if not os.path.isdir(imgs_path):
        raise ValueError(f'Nonexistent images folder: "{imgs_path}"')
        
    dataset = None
    img_names = os.listdir(imgs_path)
    n_images = len(img_names)

    with h5py.File(out_file_path, 'w') as f:
        img_ids = f.create_dataset('img_ids', (n_images,), dtype=np.int64)
        img_ids[:] = np.array([int(re.search(img_id_pattern, img_name).group(1)) for img_name in img_names])
                
        for i, img_name in tqdm(list(enumerate(img_names))):
            img_path = os.path.join(imgs_path, img_name)
            img_data = nib.load(img_path).get_fdata()

            if 'imgs' not in f:
                dataset = f.create_dataset('imgs', ((n_images,) + img_data.shape), dtype=np.float16)
            dataset[i] = img_data



# def create_cohort_hdf5(hdf5_path, nii_dir, normalize=False):
#     '''
#     hdf5_path: path to the new HDF5 file to consolidate data into
#     nii_dir: directory containing the data as nii files.
#     normalize: whether to normalize the images or not before writing
#     '''
#     imgs, img_ids, patient_info = get_normalized_data(nii_dir, normalize=normalize)

#     with h5py.File(hdf5_path, 'w') as f:
#         f.create_dataset('imgs', data=imgs)
#         f.create_dataset('img_ids', data=np.array(img_ids, dtype=int))
#         f.create_dataset('patient_info', data=np.array(patient_info, dtype='S'))
        

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
    def __init__(self, pet_hdf5_path, mri_hdf5_path, all_cohorts_path, diagnosis=3, normalize='scale', in_memory=False):
        self.pet_hdf5_path = pet_hdf5_path
        self.mri_hdf5_path = mri_hdf5_path
        self.normalize = normalize
        self.in_memory = in_memory

        self.pet_hdf5 = h5py.File(pet_hdf5_path, 'r')
        self.mri_hdf5 = h5py.File(mri_hdf5_path, 'r')

        all_cohorts_df = pd.read_csv(all_cohorts_path, index_col=0)
        self.data_df = all_cohorts_df[all_cohorts_df['DIAGNOSIS'] == diagnosis]
        self.data_df = self.data_df[(~self.data_df['CDRSB'].isna()) & (self.data_df['image_id_pet'] != 418607)]
        self.img_ids = self.data_df['image_id_pet'].to_numpy()

        if self.data_df['image_id_pet'].nunique() != len(self.data_df):
            raise ValueError(f'PET image ID column (image_id_pet) in "{all_cohorts_path}" should be unique!')

        self.pet_id_to_ind = {int(img_id): i for i, img_id in enumerate(self.pet_hdf5['img_ids'])}
        self.mri_id_to_ind = {int(img_id): i for i, img_id in enumerate(self.mri_hdf5['img_ids'])}
        
        if self.in_memory:  
            self.mri_imgs = self.mri_hdf5['imgs'][:].astype(np.float32)
            self.pet_imgs = self.pet_hdf5['imgs'][:].astype(np.float32)
            
            if self.normalize == 'scale':
                self.pet_imgs = scale_normalize(self.pet_imgs, 0, 1, batched=True)
                self.mri_imgs = scale_normalize(self.mri_imgs, 0, 1, batched=True)
                
            self.mri_imgs = torch.from_numpy(self.mri_imgs).unsqueeze(1)
            self.pet_imgs = torch.from_numpy(self.pet_imgs).unsqueeze(1)

        else:
            self.mri_imgs = self.mri_hdf5['imgs']
            self.pet_imgs = self.pet_hdf5['imgs']
        
    
    def __len__(self):
        return len(self.data_df)
    

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        pet_id, mri_id = row['image_id_pet'], row['image_id_mri']
        
        if not self.in_memory:
            pet_img = torch.from_numpy(self.pet_imgs[self.pet_id_to_ind[pet_id]].astype(np.float32)).unsqueeze(0)
            mri_img = torch.from_numpy(self.mri_imgs[self.mri_id_to_ind[mri_id]].astype(np.float32)).unsqueeze(0)

            if self.normalize == 'scale':
                pet_img = scale_normalize(pet_img, 0, 1)
                mri_img = scale_normalize(mri_img, 0, 1)
        else:
            pet_img = self.pet_imgs[self.pet_id_to_ind[pet_id]]
            mri_img = self.mri_imgs[self.mri_id_to_ind[mri_id]]

        return pet_img, mri_img, np.atleast_1d(row['CDRSB']).astype(np.float32) / CDR_MAX_SCORE


# class cVAE_MNIST(nn.Module):
#     def __init__(self, feat_dim, latent_dim, num_classes):
#         super().__init__()
        
#         # Encoder layers
#         # Provide the label as the condition
# #         self.flatten = nn.Flatten()
#         hidden_dim = 400
#         self.linear1 = nn.Linear(feat_dim+num_classes, hidden_dim)
#         self.linear2_mu = nn.Linear(hidden_dim, latent_dim)
#         self.linear2_var = nn.Linear(hidden_dim, latent_dim)
        
#         # Decoder layers
#         self.linear3 = nn.Linear(latent_dim+num_classes, hidden_dim)
#         self.linear4 = nn.Linear(hidden_dim, feat_dim)
        
#         self.sigmoid = nn.Sigmoid()
        
        
#     def encode(self, x, y):
#         '''
#         x: data input (N x feat_dim)
#         y: condition (N x num_classes)
#         '''
#         xy = torch.cat([x, y], dim=1)
#         out1 = self.linear1(xy)
#         out2 = out1.relu()
#         # Predict distribution parameters
#         z_mu = self.linear2_mu(out2)
#         z_var = self.linear2_var(out2)
#         return z_mu, z_var
    
    
#     def decode(self, z, y):
#         '''
#         z: latent features (N x latent_size)
#         y: condition (N x num_classes)
#         '''
#         zy = torch.cat([z, y], dim=1)
#         out1 = self.linear3(zy)
#         out2 = out1.relu()
#         out3 = self.linear4(out2)
#         return self.sigmoid(out3)
    
    
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(logvar/2)
#         epsilon = torch.randn_like(std)
#         return mu + epsilon*std
        
        
#     def forward(self, x, y):
#         mu, logvar = self.encode(x, y)
#         z = self.reparameterize(mu, logvar)
#         x_hat = self.decode(z, y) 
#         return x_hat, mu, logvar
    
    
    
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
    

class BimodalCVAE(nn.Module):

    def __init__(self, latent_dim, cond_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.x1_encoder = nn.Sequential(
            nn.Conv3d(1, 16, (5,5,5), stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, (5,5,5), stride=3),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )

        self.x2_encoder = nn.Sequential(
            nn.Conv3d(1, 16, (5,5,5), stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, (5,5,5), stride=3),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )

        self.fused_encoder = nn.Sequential(
            nn.Conv3d(16*2, 128, (5,5,5), stride=3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, (3,3,3)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5760, 64),
            nn.ReLU()
        )

        self.encoder_mu = nn.Linear(64+cond_dim, latent_dim)
        self.encoder_var = nn.Linear(64+cond_dim, latent_dim)

        self.fused_decoder1 = nn.Sequential(
            nn.Linear(latent_dim+cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5760),
            nn.ReLU()
        )

        self.fused_decoder2 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, (3,3,3)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 16*2, (5,5,5), stride=3, output_padding=(2,0,2)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.x1_decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 16, (5,5,5), stride=3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, (5,5,5), stride=2),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.x2_decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 16, (5,5,5), stride=3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, (5,5,5), stride=2),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )


    def encode(self, x1, x2, c):
        h1 = self.x1_encoder(x1)
        h2 = self.x2_encoder(x2)
        h = torch.cat([h1, h2], dim=1)
        h_fused = self.fused_encoder(h)

        h_c = torch.cat([h_fused, c], dim=1)
        mu = self.encoder_mu(h_c)
        logvar = self.encoder_var(h_c)
        return mu, logvar


    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2.)
        epsilon = torch.randn_like(std)
        return mu + std*epsilon
    

    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=1)
        h_fused = self.fused_decoder1(z_c)
        h_fused = h_fused.view(-1, 128, 3, 5, 3)
        h_fused = self.fused_decoder2(h_fused)

        h1, h2 = torch.split(h_fused, 16, dim=1)
        xhat1 = self.x1_decoder(h1)
        xhat2 = self.x2_decoder(h2)

        return xhat1, xhat2


    def forward(self, x1, x2, c):
        mu, logvar = self.encode(x1, x2, c)
        z = self.reparameterize(mu, logvar)
        xhat1, xhat2 = self.decode(z, c)
        return xhat1, xhat2, mu, logvar
    
    
    
    
def train_cvae(model, optimizer, dataset, loss_fn, epochs, batch_size, save_freq=None, save_path=None, scheduler=None, device='cpu'):
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
            loss, recon_loss, kl_loss = loss_fn(x, pred, mu, logvar)
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
            
        if save_freq and save_path and ((e+1) % save_freq == 0 or e == epochs-1):
            save_model(save_path, model, optimizer, epochs)
            print(f'Saved to {save_path}')       
         
#         f1 = f1_score(torch.cat(all_true, dim=0), 
#                       (torch.sigmoid(torch.cat(all_pred, dim=0)) > 0.5).type(torch.float), 
#                       average='weighted')
#         print(f'F1 score: {f1}')


def train_bimodal_cvae(model, optimizer, dataset, loss_fn, epochs, batch_size, val_dataset=None, save_freq=None, save_path=None, scheduler=None, device='cpu', verbose=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    
    validate = (val_dataset is not None)
    
    vprint = (
        print if verbose == 2 else lambda *args, **kwargs: None
    )

    progress_bar = (
        tqdm if verbose == 1 else lambda x: x
    )
    
    vprint(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    vprint(f'Scheduler: {scheduler}' if scheduler else 'No learning rate scheduling!')
    vprint(f'Training for {epochs} epochs, with batch size={batch_size}')
    vprint(f'Using device: {device}')
    vprint(f'Saving model every {save_freq} epochs to {save_path}' if save_freq else 'WARNING: Will not save model!')

    for e in progress_bar(list(range(epochs))):
        losses, recon_losses, kl_losses = [], [], []
        # all_pred, all_true = [], []
        t = time.time()
        vprint(f'\n-----Epoch {e+1}/{epochs}-----')
        for i, (pet, mri, cdr) in enumerate(loader):
            
            pet = pet.to(device)
            mri = mri.to(device)
            cdr = cdr.to(device)

            pet_hat, mri_hat, mu, logvar = model(pet, mri, cdr)
            loss, recon_loss, kl_loss = loss_fn(pet, mri, pet_hat, mri_hat, mu, logvar, invo_cvae_loss_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())

            if len(losses) == 8 or i == len(loader)-1:
                elapsed = time.time() - t
                # pred_temp = torch.cat(all_pred)
                # true_temp = torch.cat(all_true)

                vprint(f'Batch {i+1}/{len(loader)} | loss: {np.mean(losses)} ({elapsed:.3f}s) | recon: {np.mean(recon_losses)} | KL: {np.mean(kl_losses)}')
                    
                model.train()
                t = time.time()
                losses = []
                recon_losses = []
                kl_losses = []
                
#         if validate:
            # val_loss, val_acc, val_f1, val_auc = test_model(model, loss_fn, val_dataset, device=device, multiclass=multiclass)
            # vprint(f'Validation: val loss: {val_loss:.3f} | val acc: {val_acc:.3f} | val F1: {val_f1:.3f} | val AUC: {val_auc:.3f}')
            # model.train()
#             vprint()
#         else:
#             vprint()
                
        if scheduler is not None:
            scheduler.step()
            
        if save_freq and save_path and ((e+1) % save_freq == 0 or e == epochs-1):
            save_model(save_path, model, optimizer, epochs)
            vprint(f'Saved to {save_path}')

            
def bce_loss(x, x_hat):
    '''
    Binary cross-entropy loss for VAE-like loss functions that use the ELBO lower bound.
    Mean BCE across batches is returned.
    '''
#     bce_loss = F.binary_cross_entropy(x_hat, x, reduction='none')
#     return torch.mean(torch.sum(bce_loss, dim=(1,2,3,4)))
    return F.binary_cross_entropy(x_hat, x, reduction='mean')
    
    
def kld_loss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def cvae_loss_fn(x, x_hat, mu, logvar, kld_weight=1):
    
    bce = bce_loss(x, x_hat)    
#     bce_loss = F.mse_loss(x_hat, x)
#     l1 = F.binary_cross_entropy(x_hat, x, reduction='none')
#     print(torch.sum(l1, dim=(1,2,3,4)))
    kld = kld_loss(mu, logvar)
#     print(f'kld: {kld_loss}, bce: {bce_loss}')
    return bce + kld_weight*kld, bce, kld


def invo_cvae_loss_fn(x, x_hat, mu, logvar, lam=1):
    
    def reparameterize(mu, logvar):
        std = torch.exp(logvar/2.)
        epsilon = torch.randn_like(std)
        return mu + std*epsilon
    
    def pairwise_dist(X, Y):
        '''
        Pairwise distance between samples from the true posterior (Gaussian) and predicted posterior.
        That is (x[i] - y[j])^2 summed over i,j=1,..N where N = batch size.
        
        Meant to be used within a kernel, e.g. Gaussian kernel
        '''
        x_norm = (X ** 2).sum(dim=1)
        y_norm = (Y ** 2).sum(dim=1)
        dist = x_norm + y_norm - 2.0 * torch.mm(X, Y.t())
        
        return dist
    
    def mmd_loss(z_pred, z_prior):
        '''Gaussian kernel'''
        sigma_list = [1, 2, 4, 8, 16]  # multi-scale kernel
        mmd = 0.0
        
#         xx_dist = pairwise_dist(z_pred, z_pred)
#         yy_dist = pairwise_dist(z_prior, z_prior)
#         xy_dist = pairwise_dist(z_pred, z_prior)
        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma)

            XX = torch.exp(-gamma * pairwise_dist(z_pred, z_pred))
            YY = torch.exp(-gamma * pairwise_dist(z_prior, z_prior))
            XY = torch.exp(-gamma * pairwise_dist(z_pred, z_prior))

            mmd += XX.mean() + YY.mean() - 2 * XY.mean()
            
        return mmd
    
    bce = bce_loss(x, x_hat)
    z_pred = reparameterize(mu, logvar)
    z_prior = torch.randn_like(z_pred)
    mmd = mmd_loss(z_pred, z_prior)
#     print(z_pred)
    return bce + lam*mmd, bce, mmd
    
    


def bimodal_cvae_loss_fn(x1, x2, xhat1, xhat2, mu, logvar, loss_fn):
    # TODO: KLD loss is currently double counted. Fix?
    loss1, recon_loss1, kl_loss1 = loss_fn(x1, xhat1, mu, logvar)
    loss2, recon_loss2, kl_loss2 = loss_fn(x2, xhat2, mu, logvar)
    
    return loss1+loss2, recon_loss1+recon_loss2, kl_loss1+kl_loss2
#     return cvae_loss_fn(x1, xhat1, mu, logvar, kld_weight=kld_weight) + cvae_loss_fn(x2, xhat2, mu, logvar, kld_weight=kld_weight)


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


def predict_latent(model, dataset, idxs=None, device='cpu'):
    '''
    Predict the model on a given dataset, 
    '''
    chunksize = 128
    all_mu, all_logvar = [], []
    loader = DataLoader(dataset if idxs is None else Subset(dataset, idxs), batch_size=chunksize, shuffle=False)
    model.eval()
    with torch.no_grad():
        for pet, cdr in tqdm(loader):
            pet, cdr = pet.to(device), cdr.to(device)
            mu, logvar = model.encode(pet, cdr)
            all_mu.append(mu)
            all_logvar.append(logvar)
        
        return torch.cat(all_mu), torch.cat(all_logvar)


def predict_bimodal_latent(model, dataset, idxs=None, device='cpu'):
    '''
    Predict the model on a given dataset, 
    '''
    chunksize = 128
    all_mu, all_logvar = [], []
    loader = DataLoader(dataset if idxs is None else Subset(dataset, idxs), batch_size=chunksize, shuffle=False)
    model.eval()
    with torch.no_grad():
        for pet, mri, cdr in tqdm(loader):
            pet, mri, cdr = pet.to(device), mri.to(device), cdr.to(device)
            mu, logvar = model.encode(pet, mri, cdr)
            all_mu.append(mu)
            all_logvar.append(logvar)
        
        return torch.cat(all_mu), torch.cat(all_logvar)
    
    

def plot_elbows(data, max_clusters):
    '''Plot K means within-cluster sum of squares (WCSS) vs. number of clusters'''
    all_wcss = []
    dist_func = lambda x,y: (np.sum((x-y)**2))
    
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k).fit(data)
#         clusters = kmeans.cluster_centers_
#         clust_ids = kmeans.predict(data)
#         cur_wcss = 0.
#         for i in range(data.shape[0]):
#             clust_id = clust_ids[i]
#             cur_wcss += dist_func(data[i], clusters[clust_id])
        cur_wcss = kmeans.inertia_
        all_wcss.append(cur_wcss)
    
    fig, ax = plt.subplots()
    ax.plot(range(1, max_clusters+1), all_wcss)
    ax.scatter(range(1, max_clusters+1), all_wcss)
        
        
    
def plot_silhouette(X, tsne_proj, n_clusters, plot=True):
    '''
    Adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    X: (n_samples, n_features) data to cluster
    tsne_proj: (n_samples, 2) the t-SNE projection of X into 2 dimensions
    n_clusters: (int) number of clusters to consider
    plot: whether to plot the total silhouette values per cluster as well as the t-SNE plot (True) and return mean silhouette score, 
          or to just return the mean silhouette score (False).
    '''
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters)
    clust = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, clust)
    if not plot:
        return silhouette_avg
    
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    
    
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])



    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, clust)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[clust == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(clust.astype(float) / n_clusters)
    ax2.scatter(
        tsne_proj[:, 0], tsne_proj[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    # centers = tsne.transform(kmeans.cluster_centers_)
    # # Draw white circles at cluster centers
    # ax2.scatter(
    #     centers[:, 0],
    #     centers[:, 1],
    #     marker="o",
    #     c="white",
    #     alpha=1,
    #     s=200,
    #     edgecolor="k",
    # )

    # for i, c in enumerate(centers):
    #     ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )