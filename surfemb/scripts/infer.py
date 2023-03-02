import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .. import utils
from ..data import detector_crops
from ..data.config import config
from ..data import instance
from ..data.obj import load_objs
from ..data.renderer import ObjCoordRenderer
from ..surface_embedding import SurfaceEmbeddingModel
from ..pose_est import estimate_pose
from ..pose_refine import refine_pose

parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--res-data', type=int, default=256)
parser.add_argument('--res-crop', type=int, default=224)
parser.add_argument('--max-poses', type=int, default=10000)
parser.add_argument('--max-pose-evaluations', type=int, default=1000)
parser.add_argument('--no-rotation-ensemble', dest='rotation_ensemble', action='store_false')

args = parser.parse_args()
res_crop = args.res_crop
device = torch.device(args.device)
model_path = Path(args.model_path)
assert model_path.is_file()
model_name = model_path.name.split('.')[0]
dataset = model_name.split('-')[0]

results_dir = Path('data/results')
results_dir.mkdir(exist_ok=True)
poses_fp = results_dir / f'{model_name}-poses.npy'
poses_scores_fp = results_dir / f'{model_name}-poses-scores.npy'
poses_coord_scores_fp = results_dir / f'{model_name}-poses-coord-scores.npy'
poses_timings_fp = results_dir / f'{model_name}-poses-timings.npy'


print(f'this is poses_fp {poses_fp}')
for fp in poses_fp, poses_scores_fp, poses_timings_fp:
    assert not fp.exists()


# load model
model = SurfaceEmbeddingModel.load_from_checkpoint(str(model_path)).eval().to(device)  # type: SurfaceEmbeddingModel
model.freeze()

# load data
root = Path('data/bop') / dataset
cfg = config[dataset]
objs, obj_ids = load_objs(root / cfg.model_folder)
assert len(obj_ids) > 0
auxs = model.get_infer_auxs(objs=objs, crop_res=res_crop, from_detections=False)
surface_samples, surface_sample_normals = utils.load_surface_samples(dataset, obj_ids)
dataset_args = dict(dataset_root=root, obj_ids=obj_ids, auxs=auxs, cfg=cfg)
print(f"These are the dataset_args: {dataset_args}")
'''
data = detector_crops.DetectorCropDataset(
    dataset_root=root, cfg=cfg, obj_ids=obj_ids,
    detection_folder=Path(f'data/detection_results/{dataset}'),
    auxs=model.get_infer_auxs(objs=objs, crop_res=res_crop, from_detections=True)
)
'''

data = instance.BopInstanceDataset(**dataset_args, pbr=not True, test=True)
renderer = ObjCoordRenderer(objs, w=res_crop, h=res_crop)
print(f"This is the length of data {len(data)}")

# infer
all_poses = np.empty((2, len(data), 3, 4))
all_scores = np.ones(len(data)) * -np.inf
time_forward, time_pnpransac, time_refine = [], [], []
all_coord_scores = np.ones(len(data)) * -np.inf


def infer(i, d):
    obj_idx = d['obj_idx']
    img = d['rgb_crop']
    K_crop = d['K_crop']

    with utils.add_timing_to_list(time_forward):
        mask_lgts, query_img = model.infer_cnn(img, obj_idx, rotation_ensemble=args.rotation_ensemble)
        mask_lgts[0, 0].item()  # synchronize for timing

        # keys are independent of input (could be cached, but it's not the bottleneck)
        obj_ = objs[obj_idx]
        verts = surface_samples[obj_idx]
        verts_norm = (verts - obj_.offset) / obj_.scale
        obj_keys = model.infer_mlp(torch.from_numpy(verts_norm).float().to(device), obj_idx)
        verts = torch.from_numpy(verts).float().to(device)

    with utils.add_timing_to_list(time_pnpransac):
        # R, t, pose_scores, mask_scores, coord_scores,
        R_est, t_est, scores, mask_scores, coord_scores, *_ = estimate_pose(
            mask_lgts=mask_lgts, query_img=query_img,
            obj_pts=verts, obj_normals=surface_sample_normals[obj_idx], obj_keys=obj_keys,
            obj_diameter=obj_.diameter, K=K_crop,
        )
        print(f"CORRESPONDENCE SCORES {coord_scores.shape}")
        success = len(scores) > 0
        if success:
            best_idx = torch.argmax(scores).item()
            best_coord_idx = torch.argmax(coord_scores)
            all_scores[i] = scores[best_idx].item()
            all_coord_scores[i] = coord_scores[best_coord_idx].item()
            R_est, t_est = R_est[best_idx].cpu().numpy(), t_est[best_idx].cpu().numpy()[:, None]
        else:
            R_est, t_est = np.eye(3), np.zeros((3, 1))

    with utils.add_timing_to_list(time_refine):
        if success:
            R_est_r, t_est_r, score_r = refine_pose(
                R=R_est, t=t_est, query_img=query_img, K_crop=K_crop,
                renderer=renderer, obj_idx=obj_idx, obj_=obj_, model=model, keys_verts=obj_keys,
            )
        else:
            R_est_r, t_est_r = R_est, t_est

    for j, (R, t) in enumerate([(R_est, t_est), (R_est_r, t_est_r)]):
        all_poses[j, i, :3, :3] = R
        all_poses[j, i, :3, 3:] = t


for i, d in enumerate(tqdm(data, desc='running pose est.', smoothing=0)):
    infer(i, d)

time_forward = np.array(time_forward)
time_pnpransac = np.array(time_pnpransac)
time_refine = np.array(time_refine)

timings = np.stack((
    time_forward + time_pnpransac,
    time_forward + time_pnpransac + time_refine
))

np.save(str(poses_fp), all_poses)
np.save(str(poses_scores_fp), all_scores)
np.save(str(poses_coord_scores_fp), all_coord_scores)
np.save(str(poses_timings_fp), timings)
