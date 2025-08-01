import time
from options.train_options import TrainOptions
from models.networks import VGGLoss, save_checkpoint
from models.afwm import TVLoss, AFWM
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler,Subset
import random
from tensorboardX import SummaryWriter
import cv2
import datetime
from torch.optim.lr_scheduler import LambdaLR

opt = TrainOptions().parse()
path = 'runs/' + opt.name
os.makedirs(path, exist_ok=True)


def CreateDataset(opt):
    from data.cp_dataset import CPDataset
    dataset = CPDataset(opt.dataroot, mode='train', image_size=256,semantic_nc=opt.label_nc)
    # print("dataset [%s] was created" % (dataset.name()))
    # dataset.initialize(opt)
    return dataset


torch.distributed.init_process_group(backend="nccl")

os.makedirs('sample', exist_ok=True)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
# local_rank = 0
device = torch.device(f'cuda:{local_rank}')

start_epoch, epoch_iter = 1, 0

train_data_full = CreateDataset(opt)

if opt.subset_size > 0:
    print(f"Using a random subset of {opt.subset_size} images with seed {opt.random_seed}.")
    subset_size = min(opt.subset_size, len(train_data_full))
    indices = list(range(len(train_data_full)))
    random.seed(opt.random_seed)
    random.shuffle(indices)
    subset_indices = indices[:subset_size]
    train_data = Subset(train_data_full, subset_indices)
else:
    print("Using the full dataset.")
    train_data = train_data_full


train_sampler = DistributedSampler(train_data)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                          num_workers=4, pin_memory=True, sampler=train_sampler)
dataset_size = len(train_loader)
print('#training images = %d' % dataset_size)

warp_model = AFWM(opt, 3 + opt.label_nc)
# print(warp_model)
warp_model.train()
warp_model.cuda()
warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)

if opt.isTrain and len(opt.gpu_ids) > 1:
    model = torch.nn.parallel.DistributedDataParallel(warp_model, device_ids=[local_rank], output_device=local_rank)
else:
    model = warp_model

criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()

params_warp = [p for p in model.parameters()]
optimizer_warp = torch.optim.Adam(params_warp, lr=opt.lr, betas=(opt.beta1, 0.999))

scheduler = LambdaLR(
    optimizer_warp,
    lr_lambda=lambda epoch: 1.0 - max(0, epoch + opt.niter - (opt.niter + opt.niter_decay)) / float(opt.niter_decay + 1)
)

total_steps = 0 

if opt.continue_train:
    checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, f'{opt.which_epoch}.pth')
    
    if opt.continue_train:
    # Use 'latest.pth' for easy resuming, or a specific epoch file
    # If which_epoch is 'latest', use latest.pth, otherwise construct the epoch-specific path.
        if opt.which_epoch == 'latest':
            checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, 'latest.pth')
        else:
            checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, f'PBAFN_warp_epoch_{opt.which_epoch}.pth')
        
        if os.path.exists(checkpoint_path):
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # This handles both DDP and single-GPU model states
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(checkpoint['state_dict'])
            
            optimizer_warp.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            start_epoch = checkpoint['epoch'] + 1
            total_steps = checkpoint.get('step', 0)
            epoch_iter = total_steps % dataset_size

            print(f"Loaded model from epoch {checkpoint['epoch']} at step {total_steps}. Resuming from epoch {start_epoch}.")
        else:
            print(f"WARNING: --continue_train was specified, but checkpoint not found at {checkpoint_path}. Starting from scratch.")

if total_steps == 0:
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter

step = total_steps
step_per_batch = dataset_size

if local_rank == 0:
    writer = SummaryWriter(path)

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    train_sampler.set_epoch(epoch)

    for i, data in enumerate(train_loader):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1
        save_fake = True

        # input1
        c_paired = data['cloth']['paired'].cuda()
        cm_paired = data['cloth_mask']['paired']
        cm_paired = torch.FloatTensor((cm_paired.numpy() > 0.5).astype(np.float32)).cuda()
        # input2
        parse_agnostic = data['parse_agnostic'].cuda()
        densepose = data['densepose'].cuda()
        openpose = data['pose'].cuda()
        # GT
        label_onehot = data['parse_onehot'].cuda()  # CE
        label = data['parse'].cuda()  # GAN loss
        parse_cloth_mask = data['pcm'].cuda()  # L1
        im_c = data['parse_cloth'].cuda()  # VGG
        # visualization
        im = data['image']
        agnostic = data['agnostic']

        input1 = torch.cat([c_paired, cm_paired], 1)
        input2 = torch.cat([parse_agnostic, densepose], 1)

        flow_out = model(input2, c_paired, cm_paired)
        warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]

        epsilon = 0.001
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        loss_all = 0

        for num in range(5):
            cur_person_clothes = F.interpolate(im_c, scale_factor=0.5 ** (4 - num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(parse_cloth_mask, scale_factor=0.5 ** (4 - num), mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
            loss_vgg = 0
            if num >= 2:
                loss_vgg += criterionVGG(x_all[num], cur_person_clothes.cuda())
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
            b, c, h, w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x) / (b * c * h * w)
            loss_flow_y = (delta_y_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y) / (b * c * h * w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            loss_all = loss_all + (num + 1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num + 1) * 2 * loss_edge + (
                    num + 1) * 6 * loss_second_smooth

        loss_all = 0.01 * loss_smooth + loss_all

        if local_rank == 0:
            writer.add_scalar('loss_all', loss_all, step)

        optimizer_warp.zero_grad()
        loss_all.backward()
        optimizer_warp.step()
        ############## Display results and errors ##########

        path = 'sample/' + opt.name
        os.makedirs(path, exist_ok=True)
        if step % 100 == 0:
            if local_rank == 0:
                a = agnostic.cuda()
                b = im_c.cuda()
                c = c_paired.cuda()
                e = warped_cloth
                f = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
                combine = torch.cat([a[0], b[0], c[0], e[0], f[0]], 2).squeeze()
                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                writer.add_image('combine', (combine.data + 1) / 2.0, step)
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite('sample/' + opt.name + '/' + str(step) + '.jpg', bgr)

        step += 1

        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch - (step % step_per_batch)) + step_per_batch * (opt.niter + opt.niter_decay - epoch)
        eta = iter_delta_time * step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
        if step % 100 == 0:
            if local_rank == 0:
                print('{}:[epoch {}][iter {}][step {}]--[loss-{:.6f}]--[ETA-{}]'.format(now, epoch, epoch_iter, step, loss_all, eta))
        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    if local_rank == 0:
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        if local_rank == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, step))
            state_dict_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            save_path = os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_warp_epoch_%03d.pth' % epoch)
            save_checkpoint(
                model=state_dict_to_save,  # Pass the state_dict as the first argument
                optimizer=optimizer_warp,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                save_path=save_path
            )
    scheduler.step()
    # if local_rank == 0 and epoch >= opt.niter:
    # Optional: Print the new learning rate to confirm it's working
    # print(f"LR DECAY: Epoch {epoch+1}, New LR: {scheduler.get_last_lr()[0]}")