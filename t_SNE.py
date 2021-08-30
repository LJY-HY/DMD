import torch
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import argparse
from util.arguments import get_t_SNE_arguments
from util.utils import *
from dataset.build_DMD import DMD
from dataset.build_StateFarm import StateFarm
from tqdm import tqdm

def main():
    args = argparse.ArgumentParser()
    args = get_t_SNE_arguments()
    args.device = torch.device('cuda',args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # dataset/transform setting
    args.num_classes = 11

    # Get Dataloader
    if args.dataset=='StateFarm' and args.deployment_subject=='all':
        args.subject = 'all'
        del args.deployment_subject
    elif args.dataset == 'DMD':
        args.subject = 'except_'+str(args.test_subject)
        del args.deployment_subject
    else:
        args.subject = args.deployment_subject
    train_dataloader, val_dataloader, test_dataloader = globals()[args.dataset](args)

    # Get architecture
    net = get_architecture(args)
    name ='./checkpoint/'+args.arch+'_DMD_freeze_'+str(args.freeze)+'.pth'
    state_dict = torch.load(name)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    # linear drawing
    if not os.path.isdir('./t_SNE'):
        os.makedirs('./t_SNE')
    train_features_linear, y = test(args, net, train_dataloader)
    draw_tSNE(train_features_linear, y, args)
  

def test(args, net, train_dataloader):
    net.eval()
    p_bar = tqdm(range(train_dataloader.__len__()))
    train_features = torch.Tensor().to(args.device)
    y = torch.Tensor().to(args.device)
    with torch.no_grad():
        for batch_idx, loader in enumerate(train_dataloader):
            if len(loader)==3:
                images, labels, index = loader
            else:
                images, labels = loader
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = net(images, special_output = 'encoder')
            train_features = torch.cat((train_features,outputs),dim=0)
            y = torch.cat((y,labels),dim=0)
            p_bar.set_description("Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. ".format(
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=train_dataloader.__len__(),
                   ))
            p_bar.update()
    p_bar.close()
    print('forward part')
    return train_features, y

def draw_tSNE(train_features,y, args):
    train_features = train_features.cpu().numpy()
    y = y.cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=args.perplexity, n_iter=300)
    tsne_ref = tsne.fit_transform(train_features)
    df = pd.DataFrame(tsne_ref, index=tsne_ref[0:,1])
    df['x'] = tsne_ref[:,0]
    df['y'] = tsne_ref[:,1]
    df['Label'] = y[:]
    # sns.scatterplot(x="x", y="y", hue="y", palette=sns.color_palette("hls", 10), data=df)
    sns.lmplot(x="x", y="y", data=df, fit_reg=False, legend=True, size=9, hue='Label', scatter_kws={"s":200, "alpha":0.5}).savefig('t_SNE/TSNE_'+args.dataset+'_'+args.subject+'_'+str(args.perplexity)+'.png')

    plt.title('t-SNE result', weight='bold').set_fontsize('14')
    plt.xlabel('x', weight='bold').set_fontsize('10')
    plt.ylabel('y', weight='bold').set_fontsize('10')
    plt.show()

if __name__ == '__main__':
    main()