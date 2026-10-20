# Benchmark: four ways to decide the same doorbell events

10 clips, 35 events (OpenCV ran once; every method saw the same events). Decider 2B on an Apple M2 Pro laptop (MPS); Bedrock Nova Lite in us-east-1 called from the same laptop.

| Method | Visit clips alerted | Pet clips kept quiet | Phone alerts | Bedrock calls | Bedrock cost | Decision time per clip |
|---|---|---|---|---|---|---|
| rules | 7 / 7 | 0 / 3 | 20 | 0 | $0.00000 | 0.0 s |
| decider | 7 / 7 | 2 / 3 | 15 | 0 | $0.00000 | 2.5 s |
| decider+bedrock | 7 / 7 | 3 / 3 | 14 | 9 | $0.00059 | 4.3 s |
| bedrock | 4 / 7 | 3 / 3 | 4 | 35 | $0.00230 | 13.8 s |

## Per clip

| Clip | Kind | Events | rules | decider | decider+bedrock | bedrock |
|---|---|---|---|---|---|---|
| ring_7Z4ExUfkUOw | taken | 3 | A A A | A A A | A A A | i i i |
| ring_k-WvGk7a9Ms | taken | 2 | A A | i A | i A | i A |
| ring_RM8AxCNwzcQ | taken | 7 | l A i i i A A | l A i i i i A | l A i i i i A | i i i i i i i |
| ring_vS55Kxp-rZQ | taken | 3 | A i A | A i A | A i i | i i i |
| ring_nRV3nmU33oE | taken | 6 | A i i i i A | A i i i i i | A A i i i i | i A i i i i |
| ring_bkvJyN71IRE | visit | 3 | A A A | A A A | A A A | i A i |
| ring_jPQ9iD_EJEE | visit | 2 | A A | A A | A A | i A |
| ring_xEqUvf5UDnY | animal | 4 | i i i A | i i i A | i i i l | i i i l |
| pexels_dog_barking_5357485 | animal | 2 | i A | i i | i i | i i |
| pexels_dog_doorway_5394386 | animal | 3 | l l A | i i i | i i i | i i i |

Per-clip cells list each event's final action in order: **A** = alert, i = ignore, l = log, r = recheck.

