<!-- Measured at first boot on AWS EC2 t4g.xlarge (Graviton2, 4 vCPU, Ubuntu 24.04, pip OpenCV 5.0.0 with KleidiCV) by deploy/terraform/user_data.sh.tftpl; copied from CloudWatch log group /ring-clip-agent, streams <instance>/coolbench. -->

# Speed on AWS Graviton2

The OpenCV pass of the pipeline (vision/timeline.py build_timeline: decode, checks, MOG2 motion, YOLOX-S) over the
10 published clips, with no decider, timed by `vision/coolbench.py` on the demo server itself before the
services start. Both runs of a comparison happen on the same instance one after the other.

## Result (instance i-067cabfce554a0876, 2026-10-25)

`before` is the old path: frames read and YOLOX run inline, YOLOX on OpenCV 5's new dnn engine
(`PIPELINE_PARALLEL=0 DETECTOR_ENGINE=new`). `after` is the default: frames read ahead on a second thread,
YOLOX on its own thread, YOLOX on the classic dnn engine. **177.1 s -> 113.0 s, 1.57x, the same number of
events in every clip** (and identical events on all 10 clips on the M2 Pro, where the full event lists were compared).

### Speed comparison: before vs after

Same machine (aarch64, 4 CPUs). before: OpenCV 5.0.0 (KleidiCV yes); after: OpenCV 5.0.0 (KleidiCV yes). 300 real frames from the clips, median of 5 runs.

| Step | before ms | after ms | after speedup |
|---|---|---|---|
| gray + blur check (Laplacian) | 2280.47 | 2292.42 | 0.99x |
| night check (HSV) | 434.62 | 443.13 | 0.98x |
| motion (MOG2 + morphology + contours) | 7924.96 | 7923.76 | 1.00x |
| enhance (gamma LUT + CLAHE + bilateral) | 5002.23 | 4995.99 | 1.00x |
| zoom crops + letterbox | 74.09 | 74.19 | 1.00x |
| YOLOX-S detector (cv2.dnn), 10 frames | 4037.17 | 2046.89 | 1.97x |

| Whole OpenCV pass per clip | before s | after s | after speedup | events |
|---|---|---|---|---|
| pexels_dog_barking_5357485 | 11.77 | 8.43 | 1.40x | 2 / 2 |
| pexels_dog_doorway_5394386 | 22.48 | 14.37 | 1.56x | 3 / 3 |
| ring_7Z4ExUfkUOw | 26.91 | 15.33 | 1.76x | 4 / 4 |
| ring_RM8AxCNwzcQ | 19.72 | 12.29 | 1.60x | 7 / 7 |
| ring_bkvJyN71IRE | 36.48 | 20.96 | 1.74x | 4 / 4 |
| ring_jPQ9iD_EJEE | 6.23 | 4.54 | 1.37x | 2 / 2 |
| ring_k-WvGk7a9Ms | 12.33 | 9.99 | 1.23x | 2 / 2 |
| ring_nRV3nmU33oE | 21.77 | 14.76 | 1.47x | 6 / 6 |
| ring_vS55Kxp-rZQ | 7.46 | 5.13 | 1.45x | 3 / 3 |
| ring_xEqUvf5UDnY | 11.91 | 7.16 | 1.66x | 4 / 4 |
| **all clips** | **177.1** | **113.0** | **1.57x** | |

## Earlier run (instance i-04dccc5a5d225fefa, same day)

Three settings: `serial` (old path), `parallel` (threads only, new engine) and `parallel+fp16` (threads, classic
engine, `DETECTOR_FP16=1`). OpenCV logged "fall back to DNN_TARGET_CPU" for the FP16 target on Graviton2, so that
run did not use half precision, yet it was faster again (102.0 s, detector 2.65x) than the classic engine
without the setting (113.0 s, detector 1.97x, above). We report the default, not this best case.

### Speed comparison: serial vs parallel

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
### Speed comparison: serial vs parallel+fp16

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
