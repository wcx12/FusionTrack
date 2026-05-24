# MPS-GAF Schema Benchmark Results

Protocol: source-2 / crop-noise / 20 eval batches / 40 pairs; primary metric is `rot_mean_deg + 50 * trans_mean`.

| Rank | Kind | Method | Pairs | Rot mean deg | Trans mean | Chamfer mean | Pose metric | Success | Skip | Runtime s | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | learning | mps_gaf_learned_svd | 40 | 11.010597 | 0.140498 | 0.025260 | 18.035513 |  | 0.000000 |  | ours |
| 2 | learning | rpmnet | 40 | 18.043591 | 0.178193 | 0.031803 | 26.953259 |  | 0.000000 |  | RPMNet adapter |
| 3 | non_learning | icp_point_to_point | 40 | 24.867688 | 0.244304 | 0.024374 | 37.082870 | 0.475000 | 0.000000 | 0.010912 |  |
| 4 | learning | prnet_dgcnn | 40 | 24.847940 | 0.302301 | 0.028297 | 39.962994 |  | 0.000000 |  | PRNet DGCNN adapter |
| 5 | non_learning | icp_trimmed | 40 | 27.303442 | 0.274238 | 0.046286 | 41.015353 | 0.300000 | 0.000000 | 0.010877 |  |
| 6 | learning | dcp_dgcnn | 40 | 25.843506 | 0.334214 | 0.058144 | 42.554219 |  | 0.000000 |  | DCP DGCNN adapter |
| 7 | learning | pointnetlk | 40 | 30.722782 | 0.279617 | 0.047756 | 44.703648 |  | 0.000000 |  | PointNetLK adapter; emb_dims=256,n_iters=3 |
| 8 | non_learning | icp_point_to_plane | 40 | 32.139809 | 0.277369 | 0.050398 | 46.008257 | 0.525000 | 0.000000 | 0.009877 |  |
| 9 | non_learning | gicp | 40 | 31.308101 | 0.344578 | 0.130854 | 48.537012 | 0.350000 | 0.000000 | 0.005992 |  |
| 10 | learning | idam_gnn | 40 | 33.078438 | 0.346020 | 0.067264 | 50.379440 |  | 0.000000 |  | IDAM GNN adapter |
| 11 | non_learning | cpd | 40 | 39.225849 | 0.246514 | 0.020306 | 51.551562 | 0.250000 | 0.000000 | 0.508755 |  |
| 12 | learning | dcp_pointnet | 40 | 32.653061 | 0.397425 | 0.064609 | 52.524308 |  | 0.000000 |  | DCP PointNet adapter |
| 13 | non_learning | identity | 40 | 39.377473 | 0.462558 | 0.240095 | 62.505383 | 0.025000 | 0.000000 | 0.000000 |  |
| 14 | non_learning | ransac_icp | 40 | 49.230511 | 0.375405 | 0.035730 | 68.000769 | 0.225000 | 0.000000 | 0.285441 |  |
| 15 | non_learning | super4pcs | 40 | 67.134548 | 0.284804 | 0.026004 | 81.374755 | 0.500000 | 0.000000 | 2.353942 |  |
| 16 | non_learning | fpfh_fgr | 40 | 64.020661 | 0.439965 | 0.052215 | 86.018929 | 0.175000 | 0.000000 | 0.014315 |  |
| 17 | non_learning | fpfh_ransac | 40 | 91.530228 | 0.485754 | 0.059634 | 115.817916 | 0.175000 | 0.000000 | 0.266673 |  |
| 18 | non_learning | goicp | 40 | 94.764220 | 0.437973 | 0.023313 | 116.662874 | 0.200000 | 0.575000 | 3.188445 |  |
| 19 | non_learning | teaserpp | 40 | 93.299858 | 0.500926 | 0.084119 | 118.346137 | 0.175000 | 0.000000 | 0.073506 |  |
