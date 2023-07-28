from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision.utils import make_grid
from utils import *
import os
import click
import time
from itertools import chain
from network import mnist_net, generator, wideresnet
from data import data_loader
from evaluation import evaluate, evaluate_digit_save, evaluate_cifar10

HOME = os.environ['HOME']

@click.command()
@click.option('--gpu', type=str, default='0', help='gpu')
@click.option('--data', type=str, default='mnist', help='dataset')
@click.option('--tgt_epochs', type=int, default=300, help='epochs')
@click.option('--nbatch', type=int, default=100, help='每个epoch中包含多少了batch')
@click.option('--batchsize', type=int, default=256)
@click.option('--lr', type=float, default=1e-3)
@click.option('--lr_scheduler', type=str, default='none', help='scheduler')
@click.option('--ckpt', type=str, default='./saved/best.pkl')
@click.option('--num_cls', type=int, default=10, help='类别总数')
@click.option('--n_net', type=int, default=16, help='网络层数')
@click.option('--w_linear', type=float, default=0.01, help='linear weight')
@click.option('--w_info_src', type=float, default=0.1, help='info_src')
@click.option('--w_info_tgt', type=float, default=0.1, help='info_tgt')
@click.option('--w_info_g', type=float, default=0.1, help='w_info_G')
@click.option('--w_con', type=float, default=0.1, help='w_con')
@click.option('--w_noise', type=float, default=0.2, help='噪音权重')
@click.option('--feature', type=int, default=128, help='簇心特征')
@click.option('--dir', type=str, default=None, help='storage directory')
@click.option('--eval', type=bool, default=False, help='evaluation')
def experiment(gpu, data, tgt_epochs, batchsize, lr, lr_scheduler, ckpt, nbatch, \
               n_net, num_cls, w_linear, w_info_src, w_info_tgt, w_info_g, w_con, w_noise, feature, dir, eval):

    settings = locals().copy()
    print(settings)

    output_dir = make_output_dir(dir)

    model_dir = os.path.join(output_dir, 'Model')
    os.makedirs(model_dir)
    image_dir = os.path.join(output_dir, 'Image')
    os.makedirs(image_dir)
    csv_dir = os.path.join(output_dir, 'Record')
    os.makedirs(csv_dir)

    file_n = os.path.join(csv_dir, f'{data}.csv')
    file_n_best = os.path.join(csv_dir, f'{data}_best.csv')


    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    imdim = 3
    image_size = (32, 32)
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((32, 32), (0.8, 1.0)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = preprocess
    print("\n=========Building Model=========")
    print(f'train_transform {train_transform}')
    print(f'test_transform {test_transform}')

    if data == 'mnist':
        trset = data_loader.load_mnist('train', translate=train_transform)
        teset = data_loader.load_mnist('test', translate=test_transform)

        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8,sampler=RandomSampler(trset, True, nbatch * batchsize))
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
    elif data == 'cifar10':
        root_folder = './data/CIFAR-10-C'
        trset = datasets.CIFAR10(root_folder, train=True, transform=train_transform, download=True)
        teset = datasets.CIFAR10(root_folder, train=False, transform=test_transform, download=True)

        trloader = torch.utils.data.DataLoader(
            trset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=8,
            pin_memory=True, drop_last=True)
        teloader = torch.utils.data.DataLoader(
            teset,
            batch_size=batchsize,
            shuffle=False,
            num_workers=8,
            pin_memory=True, drop_last=False)

    imsize = [32, 32]

    def get_generator():

        g1_net = generator.Generator_G(n=n_net, w_noise=w_noise, imdim=imdim, imsize=imsize).cuda()
        g1_opt = optim.Adam(g1_net.parameters(), lr=lr)

        g2_net = generator.Generator_Phi(imdim=imdim, imsize=imsize).cuda()
        g2_opt = optim.Adam(g2_net.parameters(), lr=lr)

        return g1_net, g1_opt, g2_net, g2_opt

    if data == 'mnist':
        src_net = mnist_net.ConvNet().cuda()
        saved_weight = torch.load(ckpt)
        src_net.load_state_dict(saved_weight['cls_net'], strict=False)
        params = chain(src_net.parameters())
        src_opt = optim.Adam(params, lr=lr)
    elif data == 'cifar10':
        src_net = wideresnet.WideResNet(16, 10, 4, 0).cuda()
        saved_weight = torch.load(ckpt)
        src_net.load_state_dict(saved_weight['state'])
        params = chain(src_net.parameters())
        src_opt = torch.optim.SGD(
            src_net.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=0.0005,
            nesterov=True)

    #### Evaluation ####
    if eval is True:
        model_path = './models_pth/SRC_net_digits.pth'
        rst = evaluate_digit_save(gpu, model_path, file_n, batchsize=batchsize)
        print('Comparison result (in %) on digits')
        return


    centroids = nn.Linear(num_cls, feature).cuda()
    params = chain(centroids.parameters())
    optimizerC = optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))
    eye = torch.eye(num_cls).cuda()

    cls_criterion = nn.CrossEntropyLoss()
    global_best_acc = 0
    g1_net, g1_opt, g2_net, g2_opt = get_generator()
    best_acc = 0
    temp = 0.1

    #### Train #####
    for epoch in range(tgt_epochs):
        t1 = time.time()

        loss_list = []
        time_list = []
        src_net.train()

        for i, (x, y) in enumerate(trloader):
            x, y = x.cuda(), y.cuda()

            # Two generator

            # Mixing strategy

            #### Train the task model ####

            # L_info

            # L^+_info

            # L_con

            #### Trian the UDA framework ####

            # Uniformly sampling

            # min L_aug

            # max L_aug

        src_net.eval()
        if data == 'mnist':
            teacc = evaluate(src_net, teloader)
            torch.save(src_net.state_dict(), '%s/SRC_net_%d.pth' % (model_dir, epoch))
            torch.save(g1_net.state_dict(), '%s/G1_%d.pth' % (model_dir, epoch))
            pklpath = f'{model_dir}/SRC_net_{epoch}.pth'

            rst = evaluate_digit_save(gpu, pklpath, file_n, batchsize=batchsize)
            if rst[-1] >= best_acc:
                best_acc = rst[-1]
                columns = ['mnist', 'svhn', 'mnist_m', 'syndigit', 'usps', 'ave']
                df = pd.DataFrame([rst], columns=columns).to_csv(file_n_best, index=False, mode='a+', header=False)

        elif data == 'cifar10':
            if (epoch + 1) % 2 == 0:
                torch.save({'state': src_net.state_dict()}, os.path.join(model_dir, f'{epoch}_{i}-best.pkl'))
                pklpath = f'{model_dir}/{epoch}_{i}-best.pkl'
                evaluate_cifar10(gpu, pklpath, pklpath + '.test')

        l_list = []
        l_list.append(make_grid(x[0:10].detach().cpu(), 1, 2, pad_value=128))
        l_list.append(make_grid(x_aug[0:10].detach().cpu(), 1, 2, pad_value=128))
        l_list.append(make_grid(x_mix[0:10].detach().cpu(), 1, 2, pad_value=128))

        rst = make_grid(torch.stack(l_list), len(l_list), pad_value=128)
        PIL_img = transforms.ToPILImage()(rst.float())
        file_name = f'{image_dir}/im_gen_{epoch}+{i}.png'
        PIL_img.save(file_name)

        t2 = time.time()
        print(
            f'epoch {epoch}, time {t2 - t1:.2f}, ')



if __name__ == '__main__':
    manualSeed = 0
    print(f' manualSeed is {manualSeed}')
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    experiment()

