import os
import yaml
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from net_torch import FullyConvModel
import imploss
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import argparse
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import torchfilters as gf 

# ====================== ======================
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# ======================  ======================
class ImpGenSet(Dataset):
    def __init__(self, config, split='trainimp'):
        super().__init__()
        self.config = config
        self.root = config['root_dir']
        self.split = split
        self.h = config['h_dim']
        self.crop_width = 512
        self.pad_width = 500

        # 
        self.seip = sorted(list(Path(f'{self.root}/{self.split}/sx/').rglob('*.dat')))
        self.impp = sorted(list(Path(f'{self.root}/{self.split}/ws/').rglob('*.dat')))
        self.logp = sorted(list(Path(f'{self.root}/{self.split}/wx/').rglob('*.dat')))
        
        assert len(self.seip) == len(self.impp)
        assert len(self.seip) == len(self.logp)

    def __getitem__(self, idx):
        # 
        w = int(self.seip[idx].stem.split('-')[1])
        sei = np.fromfile(self.seip[idx], np.float32).reshape(1, w, self.h)
        imp = np.fromfile(self.impp[idx], np.float32).reshape(1, w, self.h)
        log = np.fromfile(self.logp[idx], np.float32).reshape(1, w, self.h)

        # mask
        mask1 = imp != 0
        mask2 = log != 0

        # 
        sei = (sei - sei.mean()) / sei.std()
        sei = sei / np.abs(sei).max()
        imp[mask1] = (imp[mask1] - imp[mask1].mean()) / imp[mask1].std()
        imp = imp / np.abs(imp).max()
        log[mask2] = (log[mask2] - log[mask2].mean()) / log[mask2].std()
        log = log / np.abs(log).max()



        w = sei.shape[1]  

        if w < self.crop_width:
                pad = self.crop_width - w
                left = pad // 2
                right = pad - left

                #
                sei = np.pad(sei,  ((0,0), (left,right), (0,0)), mode="constant", constant_values=0).astype(np.float32)
                imp = np.pad(imp,  ((0,0), (left,right), (0,0)), mode="constant", constant_values=0).astype(np.float32)
                log = np.pad(log,  ((0,0), (left,right), (0,0)), mode="constant", constant_values=0).astype(np.float32)
                mask2 = np.pad(mask2, ((0,0), (left,right), (0,0)), mode="constant", constant_values=False)

                w = self.crop_width  


        k = 0 if w == self.crop_width else random.randint(0, w - self.crop_width)

        sei = sei[:, k:k + self.crop_width, :]
        imp = imp[:, k:k + self.crop_width, :]
        log = log[:, k:k + self.crop_width, :]
        mask2 = mask2[:, k:k + self.crop_width, :]


        # 
        assert sei.shape == (1, self.crop_width, self.h)
        assert imp.shape == (1, self.crop_width, self.h)
        assert log.shape == (1, self.crop_width, self.h)

        # 
        inp = np.concatenate([sei, imp])

        return inp, mask2, log

    def __len__(self):
        return len(self.seip)

# ====================== ======================
def load_testimage(config):
    # 
    sx = np.fromfile(config['test_seis_path'], np.float32).reshape(-1, config['h_dim'])
    ws = np.fromfile(config['test_imp_path'], np.float32).reshape(-1, config['h_dim'])
    
    # 
    sx = (sx - sx.mean()) / sx.std()
    sx = sx / np.abs(sx).max()
    m1 = ws != 0
    ws[m1] = (ws[m1] - ws[m1].mean()) / ws[m1].std()
    ws = ws / np.abs(ws).max()
    
    # 
    inp = torch.from_numpy(np.concatenate([
        sx[np.newaxis, np.newaxis], 
        ws[np.newaxis, np.newaxis]
    ], 1)).to(config['device'])

    # 
    wx = np.fromfile(config['test_log_path'], np.float32).reshape(-1, config['h_dim'])
    mask = wx != 0
    loc = np.where(mask.any(axis=1))[0]
    wx = wx[loc[0]]
    mask = mask[loc[0]]
    wx[mask] = (wx[mask] - wx[mask].mean()) / wx[mask].std()
    wx = wx / np.abs(wx).max()
    
    # 
    wxf = gf.highfilter_pad(
        wx, 
        0.001, 
        10, 
        1
    )

    return inp, wx, wxf, loc[0]

# ====================== ======================
def log_image(model, inp, wx, wxf, ref, logidx, config, save_path):
    rickerf = config['rickerf']
    
    with torch.no_grad():
        out = torch.squeeze(model(inp))

    # 
    ricker = torch.from_numpy(imploss.ricker(rickerf, 0.001, 500))
    recon = imploss.ref2seis_torch(imploss.imp2ref(out), ricker)
    recon = (recon - recon.mean()) / recon.std()
    recon = recon / recon.abs().max()

    # 
    outf = gf.highfilter_pad(
        out, 
        0.001, 
        10, 
        1
    )
    
    # 
    f, a = gf.fftNd(outf, 0.001, fmax=250)
    
    # numpy
    out = out.cpu().numpy()
    outf = outf.cpu().numpy()
    ref = ref.cpu().numpy()
    seis = inp[0, 0].cpu().numpy()
    recon = recon.cpu().numpy()

    # 
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(5, 5)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    ax4 = fig.add_subplot(gs[3, :3])
    ax5 = fig.add_subplot(gs[4, :3])
    ax6 = fig.add_subplot(gs[3:, 3:])

    ax1.imshow(seis.T, aspect='auto', cmap='gray', vmin=-0.5, vmax=0.5)
    ax1.xaxis.set_visible(False)
    ax2.imshow(recon.T, aspect='auto', cmap='gray', vmin=-0.5, vmax=0.5)
    ax2.xaxis.set_visible(False)
    ax3.imshow(out.T, aspect='auto', cmap='jet', vmin=-0.6, vmax=0.6)
    ax3.xaxis.set_visible(False)
    ax4.plot(wx, label='log')
    ax4.plot(out[logidx], label='pred')
    ax4.legend()
    ax4.xaxis.set_visible(False)
    ax5.plot(wxf, label='log')
    ax5.plot(outf[logidx], label='pred')
    ax5.legend()
    ax5.xaxis.set_visible(False)
    ax6.plot(f, a, label='pred')
    ax6.plot(ref[0], ref[1], label='ref')
    
    plt.tight_layout()
    plt.savefig(
        save_path, 
        bbox_inches='tight', 
        pad_inches=0.01, 
        dpi=config['log_image_dpi']
    )
    plt.close()

# ======================  ======================
def main(config):
    # 
    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    oname = config['name'] + '_' + timestamp
    log_dir = Path(config['log_root']) / oname
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir/'models', exist_ok=True)
    os.makedirs(log_dir/'logs', exist_ok=True)
    os.makedirs(log_dir/'imgs', exist_ok=True)

    # 
    writer = SummaryWriter(log_dir=log_dir/'logs')
    writer.add_text(
        'description', 
        f"epochs: {config['epochs']}, batch_size: {config['batch_size']}, "
        f"device: {config['device']}, alpha0: {config['alpha0']}, "
        f"alpha1: {config['alpha1']}, recons_type: {config['recons_type']}"
    )

    # 
    print("Set Dataset........")
    trainset = ImpGenSet(config, 'trainimp')
    # trainset = ImpGenSet()
    trainloader = DataLoader(
        trainset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    validset = ImpGenSet(config, 'validimp')
    validloader = DataLoader(
        validset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        drop_last=True,
        num_workers=4,
    )

    # 
    print("Set model........")
    model = FullyConvModel()
    if config['pretrained']:
        params = torch.load(config['pretrained'], weights_only=True)
        model.load_state_dict(params)

    # 
    print("Set optimizer........")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [15, 25, 35, 55])

    # 
    ref = np.load(config['ref_spectrum_path'])
    rmask = np.logical_and(
        ref[0] >= 20, 
        ref[0] <= 120
    )
    ref = torch.from_numpy(ref).float().to(config['device'])
    rmask = torch.from_numpy(rmask)
    rmask = None  

    # 
    test_img, wx, wxf, logidx = load_testimage(config)
    save_img_dir = str(log_dir/'imgs')

    # 
    print("train.....")
    model = model.to(config['device'])
    global_steps = 0
    
    for epoch in range(config['epochs']):
        # 
        model.train()
        train_bar = tqdm(
            trainloader,
            desc=f"[Epoch {epoch+1}/{config['epochs']}] train",
            unit="batch",
            leave=False
        )
        
        for inp, mask, log in train_bar:
            inp = inp.to(config['device'])
            mask = mask.to(config['device'])
            log = log.to(config['device'])
            out = model(inp)

            
            l1 = F.mse_loss(out[mask], log[mask])
            
            
            if config['recons_type'] == 'mse':
                l2 = imploss.recons_loss2(
                    out, inp[:, 0:1, :, :], 
                    config['rickerf'], 0.001
                )
            elif config['recons_type'] == 'cross':
                l2 = imploss.recons_loss(
                    out, inp[:, 0:1, :, :], 
                    config['rickerf'], 0.001
                )
            elif config['recons_type'] == 'msssim':
                l2 = imploss.recons_loss3(
                    out, inp[:, 0:1, :, :], 
                    config['rickerf'], 0.001
                )
            else:
                raise ValueError(f"unknown type: {config['recons_type']}")

            
            l3 = imploss.spectrum_loss(
                out, ref[1], rmask, 
                0.001
            )

            
            loss = config['alpha0'] * l1 + config['alpha1']*l2 + config['alpha2']*l3        

            
            optim.zero_grad()
            loss.backward()
            optim.step()

            
            writer.add_scalar('train_loss', loss.item(), global_steps)
            writer.add_scalar('train_l1', l1.item(), global_steps)
            writer.add_scalar('train_l2', l2.item(), global_steps)
            writer.add_scalar('train_l3', l3.item(), global_steps)
            global_steps += 1

            train_bar.set_postfix(
                l1   = f"{l1.item():.4f}",
                l2   = f"{l2.item():.4f}",
                l3   = f"{l3.item():.4f}",
                loss = f"{loss.item():.4f}"
            )
        train_bar.close()

        
        model.eval()
        l1t, l2t, l3t, losst = [], [], [], []
        valid_bar = tqdm(
            validloader,
            desc=f"[Epoch {epoch+1}/{config['epochs']}] valid",
            unit="batch",
            leave=False
        )
        
        for inp, mask, log in valid_bar:
            with torch.no_grad():
                inp = inp.to(config['device'])
                mask = mask.to(config['device'])
                log = log.to(config['device'])
                out = model(inp)

                
                l1 = F.mse_loss(out[mask], log[mask])
                if config['recons_type'] == 'mse':
                    l2 = imploss.recons_loss2(
                        out, inp[:, 0:1, :, :], 
                        config['rickerf'], 0.001
                    )
                elif config['recons_type'] == 'cross':
                    l2 = imploss.recons_loss(
                        out, inp[:, 0:1, :, :], 
                        config['rickerf'], 0.001
                    )
                elif config['recons_type'] == 'msssim':
                    l2 = imploss.recons_loss3(
                        out, inp[:, 0:1, :, :], 
                        config['rickerf'], 0.001
                    )
                l3 = imploss.spectrum_loss(
                    out, ref[1], rmask, 
                    0.001
                )
                loss = config['alpha0'] * l1 + config['alpha1'] * l2 + config['alpha2'] * l3

                l1t.append(l1.item())
                l2t.append(l2.item())
                l3t.append(l3.item())
                losst.append(loss.item())

                valid_bar.set_postfix(
                    l1   = f"{l1.item():.4f}",
                    l2   = f"{l2.item():.4f}",
                    l3   = f"{l3.item():.4f}",
                    loss = f"{loss.item():.4f}"
                )
        valid_bar.close()

        
        l1_mean = np.array(l1t).mean()
        l2_mean = np.array(l2t).mean()
        l3_mean = np.array(l3t).mean()
        loss_mean = np.array(losst).mean()
        
        writer.add_scalar('valid_loss', loss_mean, epoch)
        writer.add_scalar('valid_l1', l1_mean, epoch)
        writer.add_scalar('valid_l2', l2_mean, epoch)
        writer.add_scalar('valid_l3', l3_mean, epoch)

        
        log_image(model, test_img, wx, wxf, ref, logidx, config, f"{save_img_dir}/{epoch}.png")

        
        scheduler.step()

        
        if epoch % config['save_model_interval'] == 0:
            print(f'save model (epoch = {epoch})')
            torch.save(model.state_dict(), log_dir/'models'/f'model_{epoch}.pt')

    
    writer.close()
    print(f'save model (epoch = last)')
    torch.save(model.state_dict(), log_dir/'models'/f'model_last.pt')

# ===================== ======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='impedance inversion')
    parser.add_argument('--config', type=str, default='train_config.yaml', 
                        help='Path to training config file (YAML)')
    args = parser.parse_args()
    
    # 
    config = load_config(args.config)
    

    main(config)