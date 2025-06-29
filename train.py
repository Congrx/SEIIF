import time
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils import *
from einops import rearrange
from model_InterNet import Net
from imresize import *
from math import sqrt
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument('--model_name', type=str, default='InterNet')
    parser.add_argument('--trainset_dir_2', type=str, default='datasets/train/SR_5x5_2x/')
    parser.add_argument('--trainset_dir_3', type=str, default='datasets/train/SR_5x5_3x/')
    parser.add_argument('--trainset_dir_4', type=str, default='datasets/train/SR_5x5_4x/')
    parser.add_argument('--testset_dir', type=str, default='datasets/test/SR_5x5_2x/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--patchsize", type=int, default=64, help="crop into patches for validation")
    parser.add_argument("--stride", type=int, default=32, help="stride for patch cropping")
    
    
    # continuous-scale training
    parser.add_argument('--train_stage', type=int, default=1, help='1 means continuous-scale training and 2 means fine-tuning fixed-scale training')
    parser.add_argument("--finetune_factor", type=int, default=2, help="upscale factor for fine-tuning")
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=101, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=20, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='.')
    
    # fine-tuning fixed-scale training
    '''
    parser.add_argument('--train_stage', type=int, default=2, help='1 means continuous-scale training and 2 means fine-tuning fixed-scale training')
    parser.add_argument("--finetune_factor", type=int, default=2, help="upscale factor for fine-tuning")
    parser.add_argument('--lr', type=float, default=2.5e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=111, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=5, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--model_path', type=str, default='./log/InterNet_ArbitrarySR_5x5_epoch_18.pth.tar')
    '''

    return parser.parse_args()


def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st

# B N^2 C H W -> B C NH NW
def FormOutput(intra_fea):
    b, n, c, h, w = intra_fea.shape
    angRes = int(sqrt(n+1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(intra_fea[:, kk, :, :, :])
            kk = kk+1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out


def feed_data(label,scale,cfg):
    label_HR = LFsplit(label,cfg.angRes)  
    h_size = round(32*scale)
    w_size = round(32*scale)    
    b,n,c,h,w = label_HR.shape
    label_LR_final = []
    label_HR_final = []
    for i in range(b):      
        h_start = random.randint(0, h - h_size)
        w_start = random.randint(0, w - w_size)
        label_HR_item = label_HR[i,:,:,h_start:h_start + h_size, w_start:w_start + w_size]  # 25 * 1 * 32s * 32s
        n,c,h,w = label_HR_item.shape
        label_HR_final.append(label_HR_item)
        label_HR_item = rearrange(label_HR_item,'N C H W -> H W (N C)')
        label_LR_item = imresize(label_HR_item, scalar_scale=1/scale)
        label_LR_item = torch.tensor(np.float32(label_LR_item))
        label_LR_final.append(rearrange(label_LR_item,'H W (N C) -> N C H W',N=n,C=c))
    label_LR_final = torch.stack(label_LR_final,dim=0)
    label_HR_final = torch.stack(label_HR_final,dim=0)
    return FormOutput(label_LR_final),FormOutput(label_HR_final),h_size,w_size,scale


def train(cfg, train_sets, test_Names, test_loaders):

    net = Net(cfg.angRes)
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            try:
                new_state_dict = OrderedDict()
                for k, v in model['state_dict'].items():
                    name = k[7:]  # add `module.`
                    new_state_dict[name] = v
                # load params
                net.load_state_dict(new_state_dict)
                print('Use pretrain model!')
            except:
                new_state_dict = OrderedDict()
                for k, v in model['state_dict'].items():
                    new_state_dict[k] = v
                # load params
                net.load_state_dict(new_state_dict)
                print('Use pretrain model!')
                pass
            pass
            epoch_state = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.model_path))


    net = torch.nn.DataParallel(net, device_ids=[0, 1])

    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler._step_count = epoch_state
    loss_epoch = []
    loss_list = []

    
    train_loader_2 = DataLoader(dataset=train_sets[0], num_workers=2, batch_size=cfg.batch_size, shuffle=True)
    train_loader_3 = DataLoader(dataset=train_sets[1], num_workers=2, batch_size=cfg.batch_size, shuffle=True)
    train_loader_4 = DataLoader(dataset=train_sets[2], num_workers=2, batch_size=cfg.batch_size, shuffle=True)
    total_1 = len(train_loader_2)
    total_2 = len(train_loader_3)
    total_3 = len(train_loader_4)
    total = min(total_1,total_2,total_3) * 6    
    data_iter_2 = iter(train_loader_2)
    data_iter_3 = iter(train_loader_3)
    data_iter_4 = iter(train_loader_4)
    for idx_epoch in range(epoch_state, cfg.n_epochs):
        if cfg.train_stage == 1:
            for idx_iter in tqdm(range(total)):
                random_choice = random.randint(1, 6)
                total_scale = [1.5,2,2.5,3,3.5,4]
                scale = total_scale[random_choice-1]
                if scale <= 2:
                    try:
                        data,label = next(data_iter_2)
                    except StopIteration:
                        data_iter_2 = iter(DataLoader(dataset=train_sets[0], num_workers=2, batch_size=cfg.batch_size, shuffle=True))
                        data,label = next(data_iter_2)
                elif scale <= 3:
                    try:
                        data,label = next(data_iter_3)
                    except StopIteration:
                        data_iter_3 = iter(DataLoader(dataset=train_sets[1], num_workers=2, batch_size=cfg.batch_size, shuffle=True))
                        data,label = next(data_iter_3)
                else:
                    try:
                        data,label = next(data_iter_4)
                    except StopIteration:
                        data_iter_4 = iter(DataLoader(dataset=train_sets[2], num_workers=2, batch_size=cfg.batch_size, shuffle=True))
                        data,label = next(data_iter_4)
                data,label,h_size,w_size,scale = feed_data(label,scale,cfg)     
                data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
                out = net(data,h_size,w_size,scale)
                loss = criterion_Loss(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch.append(loss.data.cpu())
        else:
            if cfg.finetune_factor==2:
                for idx_iter in tqdm(range(total_1)):
                    scale = 2
                    try:
                        data,label = next(data_iter_2)
                    except StopIteration:
                        data_iter_2 = iter(DataLoader(dataset=train_sets[0], num_workers=2, batch_size=cfg.batch_size, shuffle=True))
                        data,label = next(data_iter_2)
                    data,label,h_size,w_size,scale = feed_data(label,scale,cfg)    
                    data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
                    out = net(data,h_size,w_size,scale)
                    loss = criterion_Loss(out, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_epoch.append(loss.data.cpu())
            elif cfg.finetune_factor==3:
                for idx_iter in tqdm(range(total_2)):
                    scale = 3
                    try:
                        data,label = next(data_iter_3)
                    except StopIteration:
                        data_iter_3 = iter(DataLoader(dataset=train_sets[1], num_workers=2, batch_size=cfg.batch_size, shuffle=True))
                        data,label = next(data_iter_3)
                    data,label,h_size,w_size,scale = feed_data(label,scale,cfg)   
                    data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
                    out = net(data,h_size,w_size,scale)
                    loss = criterion_Loss(out, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_epoch.append(loss.data.cpu())
            elif cfg.finetune_factor==4:
                for idx_iter in tqdm(range(total_3)):
                    scale = 4
                    try:
                        data,label = next(data_iter_4)
                    except StopIteration:
                        data_iter_4 = iter(DataLoader(dataset=train_sets[2], num_workers=2, batch_size=cfg.batch_size, shuffle=True))
                        data,label = next(data_iter_4)
                    data,label,h_size,w_size,scale = feed_data(label,scale,cfg)   
                    data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
                    out = net(data,h_size,w_size,scale)
                    loss = criterion_Loss(out, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_epoch.append(loss.data.cpu())
            
        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            if cfg.train_stage == 2:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'loss': loss_list,},
                    save_path='./log/', filename=cfg.model_name + '_' + str(cfg.finetune_factor) + 'xSR_' + str(cfg.angRes) +
                                'x' + str(cfg.angRes) + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            else:
                save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'loss': loss_list,},
                    save_path='./log/', filename=cfg.model_name + '_' + 'ArbitrarySR_' + str(cfg.angRes) +
                                'x' + str(cfg.angRes) + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []
        
        ''' evaluation '''
        
        if idx_epoch % 5 == 0:
            with torch.no_grad():
                net.eval()
                psnr_testset = []
                ssim_testset = []
                for index, test_name in enumerate(test_Names):
                    test_loader = test_loaders[index]
                    psnr_epoch_test, ssim_epoch_test = valid(test_loader, net)
                    psnr_testset.append(psnr_epoch_test)
                    ssim_testset.append(ssim_epoch_test)
                    print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
                    pass
                pass
                print('Average: PSNR---%f, SSIM---%f' %(sum(psnr_testset)/5,sum(ssim_testset)/5))

        scheduler.step()
        pass
        

def valid(test_loader, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze()

        uh, vw = data.shape
        h0, w0 = uh // cfg.angRes, vw // cfg.angRes
        subLFin = LFdivide(data, cfg.angRes, cfg.patchsize, cfg.stride)  # numU, numV, h*angRes, w*angRes
        numU, numV, H, W = subLFin.shape
        subLFout = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize * cfg.finetune_factor, cfg.angRes * cfg.patchsize * cfg.finetune_factor)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                _,_,h_input,w_input = tmp.shape
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    out = net(tmp.to(cfg.device),h_input*cfg.finetune_factor//cfg.angRes,w_input*cfg.finetune_factor//cfg.angRes,cfg.finetune_factor)
                    subLFout[u, v, :, :] = out.squeeze()

        outLF = LFintegrate(subLFout, cfg.angRes, cfg.patchsize * cfg.finetune_factor, cfg.stride * cfg.finetune_factor, h0 * cfg.finetune_factor, w0 * cfg.finetune_factor)

        psnr, ssim = cal_metrics(label, outLF, cfg.angRes)

        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))


def main(cfg):
    train_set_2 = TrainSetLoader(dataset_dir=cfg.trainset_dir_2)
    train_set_3 = TrainSetLoader(dataset_dir=cfg.trainset_dir_3)
    train_set_4 = TrainSetLoader(dataset_dir=cfg.trainset_dir_4)
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    train(cfg, [train_set_2,train_set_3,train_set_4], test_Names, test_Loaders)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
