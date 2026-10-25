<!-- AWS EC2 t4g.xlarge (Graviton2, 4 vCPU, Ubuntu 24.04), instance i-04dccc5a5d225fefa, 2026-10-25; measured at first boot by deploy/terraform/user_data.sh.tftpl, copied from CloudWatch (/ring-clip-agent, stream i-04dccc5a5d225fefa/coolbench). -->

# Speed on AWS Graviton2

`serial` is the old path (frames and YOLOX inline, OpenCV 5's new dnn engine). `parallel` reads frames and runs YOLOX alongside the analysis. `parallel+fp16` also loads YOLOX on the classic dnn engine with an FP16 target; Graviton2 has no FP16 path in OpenCV's dnn and logs "fall back to DNN_TARGET_CPU", so that run is the classic engine in FP32. The 2.65x on the detector is the classic engine, which is now the default (DETECTOR_ENGINE=classic). Event counts match in every clip.

## Speed comparison: serial vs parallel

Same machine (aarch64, 4 CPUs). serial: OpenCV 5.0.0 (KleidiCV yes); parallel: OpenCV 5.0.0 (KleidiCV yes). 300 real frames from the clips, median of 5 runs.

| Step | serial ms | parallel ms | parallel speedup |
|---|---|---|---|
| gray + blur check (Laplacian) | 2279.95 | 2262.43 | 1.01x |
| night check (HSV) | 444.56 | 439.8 | 1.01x |
| motion (MOG2 + morphology + contours) | 7987.83 | 7912.72 | 1.01x |
| enhance (gamma LUT + CLAHE + bilateral) | 5056.71 | 4943.31 | 1.02x |
| zoom crops + letterbox | 74.88 | 74.05 | 1.01x |
| YOLOX-S detector (cv2.dnn), 10 frames | 4080.45 | 4067.99 | 1.00x |

| Whole OpenCV pass per clip | serial s | parallel s | parallel speedup | events |
|---|---|---|---|---|
| pexels_dog_barking_5357485 | 11.57 | 9.25 | 1.25x | 2 / 2 |
| pexels_dog_doorway_5394386 | 22.25 | 20.82 | 1.07x | 3 / 3 |
| ring_7Z4ExUfkUOw | 27.52 | 23.07 | 1.19x | 4 / 4 |
| ring_RM8AxCNwzcQ | 19.04 | 15.24 | 1.25x | 7 / 7 |
| ring_bkvJyN71IRE | 36.8 | 33.41 | 1.10x | 4 / 4 |
| ring_jPQ9iD_EJEE | 6.06 | 5.4 | 1.12x | 2 / 2 |
| ring_k-WvGk7a9Ms | 12.16 | 10.41 | 1.17x | 2 / 2 |
| ring_nRV3nmU33oE | 21.43 | 16.68 | 1.28x | 6 / 6 |
| ring_vS55Kxp-rZQ | 7.65 | 5.94 | 1.29x | 3 / 3 |
| ring_xEqUvf5UDnY | 12.35 | 9.88 | 1.25x | 4 / 4 |
| **all clips** | **176.8** | **150.1** | **1.18x** | |
## Speed comparison: serial vs parallel+fp16

Same machine (aarch64, 4 CPUs). serial: OpenCV 5.0.0 (KleidiCV yes); parallel+fp16: OpenCV 5.0.0 (KleidiCV yes). 300 real frames from the clips, median of 5 runs.

| Step | serial ms | parallel+fp16 ms | parallel+fp16 speedup |
|---|---|---|---|
| gray + blur check (Laplacian) | 2279.95 | 2251.84 | 1.01x |
| night check (HSV) | 444.56 | 438.91 | 1.01x |
| motion (MOG2 + morphology + contours) | 7987.83 | 7914.02 | 1.01x |
| enhance (gamma LUT + CLAHE + bilateral) | 5056.71 | 4966.97 | 1.02x |
| zoom crops + letterbox | 74.88 | 73.86 | 1.01x |
| YOLOX-S detector (cv2.dnn), 10 frames | 4080.45 | 1539.99 | 2.65x |

| Whole OpenCV pass per clip | serial s | parallel+fp16 s | parallel+fp16 speedup | events |
|---|---|---|---|---|
| pexels_dog_barking_5357485 | 11.57 | 7.82 | 1.48x | 2 / 2 |
| pexels_dog_doorway_5394386 | 22.25 | 12.78 | 1.74x | 3 / 3 |
| ring_7Z4ExUfkUOw | 27.52 | 12.91 | 2.13x | 4 / 4 |
| ring_RM8AxCNwzcQ | 19.04 | 10.96 | 1.74x | 7 / 7 |
| ring_bkvJyN71IRE | 36.8 | 17.79 | 2.07x | 4 / 4 |
| ring_jPQ9iD_EJEE | 6.06 | 4.19 | 1.45x | 2 / 2 |
| ring_k-WvGk7a9Ms | 12.16 | 10.01 | 1.21x | 2 / 2 |
| ring_nRV3nmU33oE | 21.43 | 13.84 | 1.55x | 6 / 6 |
| ring_vS55Kxp-rZQ | 7.65 | 5.14 | 1.49x | 3 / 3 |
| ring_xEqUvf5UDnY | 12.35 | 6.58 | 1.88x | 4 / 4 |
| **all clips** | **176.8** | **102.0** | **1.73x** | |
