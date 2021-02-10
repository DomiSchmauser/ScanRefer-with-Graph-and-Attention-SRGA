import os
import sys
import json
import pickle
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

#sys.path.append("C:/Users/User/Desktop/Semester2/ADL4CV/3d_Visual_Grounding_with_Graph_and_Attention/Implementation/ScanRefer") # HACK add the root folder
sys.path.append(os.path.join(os.getcwd()))

from lib.config import CONF
from lib.dataset import LangDataset
from lib.solver import Solver
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from models.refnet import RefNet
from data.scannet.model_util_scannet import ScannetDatasetConfig

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))


def toHex(n):
    n = int(n)
    n = 255-n
    return hex(n)[2:].zfill(2).upper()


def get_dataloader(args, scanrefer, all_scene_list, split, config):
    dataset = LangDataset(
        scanrefer=scanrefer,
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_color=args.use_color,
        use_height=(not args.no_height),
        use_normal=args.use_normal,
        use_multiview=args.use_multiview
    )
    print("evaluate on {} samples".format(len(dataset)))


    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    return dataset, dataloader

def get_model(args, config):
    # load model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = RefNet(
        num_class=config.num_class,
        num_heading_bin=config.num_heading_bin,
        num_size_cluster=config.num_size_cluster,
        mean_size_arr=config.mean_size_arr,
        num_proposal=args.num_proposals,
        input_feature_dim=input_channels,
        use_lang_classifier=True,
        use_bidir=args.use_bidir
    ).cuda()

    model_name =  "model.pth"
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(args):

    scanrefer = SCANREFER_TRAIN if args.use_train else SCANREFER_VAL
    scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
    if args.num_scenes != -1:
        scene_list = scene_list[:args.num_scenes]

    scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

    return scanrefer, scene_list



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print("Visualising attention weights...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    scanrefer, scene_list = get_scanrefer(args)

    # model
    model = get_model(args, DC)
    print(model.lang)

    # dataloader
    dataset, loader = get_dataloader(args, scanrefer, scene_list, "val", DC)

    for data in tqdm(loader):
        for key in data:
            if key != 'token_list':
                data[key] = data[key].cuda()

        with torch.no_grad():
            data = model.lang(data)
            attention = torch.sum(data["attention_weights"], 1)
            html = str()
            for bi in range(1):
                norm_attention = ((attention[bi] - torch.min(attention[bi])) / (
                            torch.max(attention[bi]) - torch.min(attention[bi]))).tolist()

                text = data["token_list"]

                for ti, ai in zip(text, norm_attention):
                    html += """<span style="background-color: #FF%s%s">%s </span>""" % (
                    toHex(ai * 255), toHex(ai * 255), ti)

                html += """</br></br>"""


        with open(os.path.join('outputs', args.folder, 'attn_weights.html'), 'a') as fout:
                fout.write(html)

        fout.close()

