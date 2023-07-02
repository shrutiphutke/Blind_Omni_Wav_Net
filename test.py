import os

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_directional_query_od import Inpainting
from utils import parse_args, TestDataset
from thop import profile
from ptflops import get_model_complexity_info

def test_loop(net, data_loader, num_iter):
    net.eval()
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for corrupted, target, name, h, w in test_bar:
            corrupted, target, = corrupted.cuda(), target.cuda()
            out = torch.clamp((torch.clamp(model(corrupted)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            target = torch.clamp(target[:, :, :h, :w].mul(255), 0, 255).byte()

            dataset_name = 'results/{}/{}'.format(args.dataset_name, name[0])
            if not os.path.exists(os.path.dirname(dataset_name)):
                os.makedirs(os.path.dirname(dataset_name))
            Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save(dataset_name)
            test_bar.set_description('Test Iter: [{}/{}]'
                                     .format(num_iter, 1 if args.model_file else args.num_iter))

if __name__ == '__main__':
    args = parse_args()
    test_dataset = TestDataset(args.data_path_test, args.data_path_test, args.task_name, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    model = Inpainting(num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48//3, 96//3, 192//3, 384//3], num_refinement=4, expansion_factor=2.66).cuda()

    model.load_state_dict(torch.load(args.model_file))
    test_loop(model, test_loader, 1)