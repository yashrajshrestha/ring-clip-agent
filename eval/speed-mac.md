<!-- Apple M2 Pro laptop, 2026-10-24, before the classic dnn engine became the default (both runs below use the new engine; the classic engine adds 1.03x here and 2.65x on Graviton2, see eval/speed-graviton.md). -->

Before these changes (commit 433706d) the same OpenCV pass over the 10 clips took 46.2 s (mean of 2 runs); the switch to OpenCV-native counting alone brings it to 42.1 s with identical events. The tables below start from there.

# Speed comparison: serial vs parallel

Same machine (arm64, 10 CPUs). serial: OpenCV 5.0.0 (KleidiCV yes); parallel: OpenCV 5.0.0 (KleidiCV yes). 300 real frames from the clips, median of 5 runs.

| Step | serial ms | parallel ms | parallel speedup |
|---|---|---|---|
| gray + blur check (Laplacian) | 520.28 | 519.8 | 1.00x |
| night check (HSV) | 217.73 | 218.09 | 1.00x |
| motion (MOG2 + morphology + contours) | 2193.35 | 2185.42 | 1.00x |
| enhance (gamma LUT + CLAHE + bilateral) | 1304.27 | 1289.03 | 1.01x |
| zoom crops + letterbox | 26.17 | 26.55 | 0.99x |
| YOLOX-S detector (cv2.dnn), 10 frames | 636.89 | 659.97 | 0.97x |

| Whole OpenCV pass per clip | serial s | parallel s | parallel speedup | events |
|---|---|---|---|---|
| ring_7Z4ExUfkUOw | 5.41 | 4.14 | 1.31x | 3 / 3 |
| ring_k-WvGk7a9Ms | 3.66 | 3.28 | 1.12x | 2 / 2 |
| ring_RM8AxCNwzcQ | 4.72 | 3.85 | 1.23x | 7 / 7 |
| ring_vS55Kxp-rZQ | 1.98 | 1.66 | 1.19x | 3 / 3 |
| ring_nRV3nmU33oE | 5.73 | 4.82 | 1.19x | 6 / 6 |
| ring_bkvJyN71IRE | 8.44 | 6.69 | 1.26x | 3 / 3 |
| ring_jPQ9iD_EJEE | 1.68 | 1.42 | 1.18x | 2 / 2 |
| ring_xEqUvf5UDnY | 2.86 | 2.28 | 1.25x | 4 / 4 |
| pexels_dog_barking_5357485 | 3.09 | 2.66 | 1.16x | 2 / 2 |
| pexels_dog_doorway_5394386 | 4.67 | 4.17 | 1.12x | 3 / 3 |
| **all clips** | **42.2** | **35.0** | **1.21x** | |


| Step | serial ms | parallel+fp16 ms | parallel+fp16 speedup |
|---|---|---|---|
| gray + blur check (Laplacian) | 520.28 | 521.07 | 1.00x |
| night check (HSV) | 217.73 | 201.58 | 1.08x |
| motion (MOG2 + morphology + contours) | 2193.35 | 2234.26 | 0.98x |
| enhance (gamma LUT + CLAHE + bilateral) | 1304.27 | 1310.03 | 1.00x |
| zoom crops + letterbox | 26.17 | 26.3 | 1.00x |
| YOLOX-S detector (cv2.dnn), 10 frames | 636.89 | 560.06 | 1.14x |

| Whole OpenCV pass per clip | serial s | parallel+fp16 s | parallel+fp16 speedup | events |
|---|---|---|---|---|
| ring_7Z4ExUfkUOw | 5.41 | 3.66 | 1.48x | 3 / 3 |
| ring_k-WvGk7a9Ms | 3.66 | 3.34 | 1.10x | 2 / 2 |
| ring_RM8AxCNwzcQ | 4.72 | 3.4 | 1.39x | 7 / 7 |
| ring_vS55Kxp-rZQ | 1.98 | 1.67 | 1.19x | 3 / 3 |
| ring_nRV3nmU33oE | 5.73 | 4.48 | 1.28x | 6 / 6 |
| ring_bkvJyN71IRE | 8.44 | 5.58 | 1.51x | 3 / 3 |
| ring_jPQ9iD_EJEE | 1.68 | 1.28 | 1.31x | 2 / 2 |
| ring_xEqUvf5UDnY | 2.86 | 2.07 | 1.38x | 4 / 4 |
| pexels_dog_barking_5357485 | 3.09 | 2.46 | 1.26x | 2 / 2 |
| pexels_dog_doorway_5394386 | 4.67 | 3.66 | 1.28x | 3 / 3 |
| **all clips** | **42.2** | **31.6** | **1.34x** | |
