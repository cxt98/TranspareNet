import torch
import open3d as o3d
import numpy as np
from grnet_point_cloud_completion.datasets.img2pcd import img2pcdHelper
from grnet_point_cloud_completion.datasets.pcd2img import pcd2imgHelper


def inference_pcc(grnet, n_points, K, batch):
    mask = batch["mask"][0].cpu().detach().numpy()[0]
    depth = batch["raw_depth"][0].cpu().detach().numpy()[0]
    # Switch models to evaluation mode
    grnet.eval()
    INV_K = np.linalg.inv(K)
    maxdis, pcds, centers = img2pcdHelper(mask, depth, INV_K)
    complete_pcds = []
    for i in range(len(pcds)):
        ptcloud = pcds[i]
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:n_points]]

        if ptcloud.shape[0] < n_points:
            zeros = np.zeros((n_points - ptcloud.shape[0], 3))
            # ptcloud = np.concatenate([ptcloud, zeros])
        ptcloud = torch.from_numpy(ptcloud.copy()).float().cuda().unsqueeze(0)
        data = {'partial_cloud': ptcloud}
        try:
            sparse_ptcloud, dense_ptcloud = grnet(data)
            dense_ptcloud = dense_ptcloud.squeeze().cpu().detach().numpy()
        except:
            dense_ptcloud = ptcloud.squeeze().cpu().detach().numpy()

        complete_pcds.append(dense_ptcloud)
    pred_depth = pcd2imgHelper(mask, depth, K, complete_pcds, maxdis, centers)
    batch["raw_depth"][0] = torch.from_numpy(pred_depth.copy()).float().cuda()
    # visualize point cloud difference through GRNet
    # pcd_before = o3d.geometry.PointCloud()
    # pcd_before.points = o3d.utility.Vector3dVector(depth2xyz(depth, K))
    # pcd_before.paint_uniform_color([0, 0, 1])
    # pcd_after  = o3d.geometry.PointCloud()
    # pcd_after.points  = o3d.utility.Vector3dVector(depth2xyz(pred_depth, K))
    # pcd_before.paint_uniform_color([0, 1, 1])
    # o3d.visualization.draw_geometries([pcd_before])#, pcd_after])
    return batch

def depth2xyz(depth, K):
    w, h = depth.shape
    u, v = np.meshgrid(np.array(range(h)), np.array(range(w)))
    xyz = np.einsum('ij,jlk->ilk', np.linalg.inv(K), np.stack((u, v, np.ones_like(u)))) * depth
    return xyz.reshape(-1, 3)