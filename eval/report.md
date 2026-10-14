# Evaluation report

18 labelled clips (every one a delivery or a theft, so each should alert at least once and detect an object left or taken).

## Summary

| Measure | Result |
|---|---|
| Clips with at least one alert | 18 / 18 |
| Events that would wake a language model (alerts) | 92 / 132 |
| Object left/taken detected | 11 / 18 |
| Object left/taken detected at night | 3 / 4 |
| Alerts per clip, median (max) | 4 (14) |
| Blurry/dark events | 0 |
| Rechecks, of which rescued by enhancement | 0, 0 |
| Decision sources | strands-decider-2b: 132 |

## Per clip

| Clip | Kind | Night | Events | Alerts | Person | Left/taken | Quality | Cuts | Result |
|---|---|---|---|---|---|---|---|---|---|
| amazon_d2yRFg-Xfqs | left |  | 13 | 9 | 4 | 2 | 0 | 4 | ok |
| amazon__iV2anL2SKE | left |  | 11 | 7 | 5 | 1 | 0 | 4 | ok |
| amazon_O_qJFsKEFmQ | left |  | 10 | 9 | 8 | 1 | 0 | 1 | ok |
| amazon_miRKZwhIA70 | left |  | 17 | 14 | 14 | 0 | 0 | 3 | no left/taken event |
| delivery_OgmMGQ_OYrY | left | yes | 1 | 1 | 0 | 0 | 0 | 0 | no left/taken event |
| delivery_Xl-R3YCQRkY | left |  | 5 | 3 | 3 | 0 | 0 | 1 | no left/taken event |
| delivery_H1bjA7jJC4Q | left |  | 2 | 2 | 1 | 1 | 0 | 0 | ok |
| delivery_D59d7IwumxI | left |  | 2 | 2 | 1 | 1 | 0 | 0 | ok |
| delivery__xg3SH6qu1c | left |  | 21 | 9 | 8 | 1 | 0 | 8 | ok |
| delivery_sBnub4-N-lk | left |  | 11 | 8 | 8 | 0 | 0 | 3 | no left/taken event |
| theft_7Z4ExUfkUOw | taken |  | 4 | 4 | 4 | 0 | 0 | 0 | no left/taken event |
| theft_k-WvGk7a9Ms | taken | yes | 2 | 2 | 0 | 1 | 0 | 0 | ok |
| theft_luvoxnxVBKA | taken | yes | 5 | 5 | 3 | 1 | 0 | 0 | ok |
| theft_QKocOIxfqu8 | taken |  | 7 | 4 | 4 | 0 | 0 | 3 | no left/taken event |
| theft_2K2sh8xHyIA | taken |  | 5 | 4 | 1 | 2 | 0 | 1 | ok |
| theft_RM8AxCNwzcQ | taken |  | 8 | 3 | 2 | 1 | 0 | 1 | ok |
| theft_r1cmcLUahKk | taken |  | 5 | 4 | 4 | 0 | 0 | 1 | no left/taken event |
| theft_vS55Kxp-rZQ | taken | yes | 3 | 2 | 1 | 1 | 0 | 0 | ok |

## Failure cases

- **amazon_miRKZwhIA70** (left): the package being left or taken was not detected. compilation of 3 visits; third repeats O_qJFsKEFmQ
- **delivery_OgmMGQ_OYrY** (left, night): the package being left or taken was not detected. late-night delivery
- **delivery_Xl-R3YCQRkY** (left): the package being left or taken was not detected. would-be thief drops the package
- **delivery_sBnub4-N-lk** (left): the package being left or taken was not detected. many boxes
- **theft_7Z4ExUfkUOw** (taken): the package being left or taken was not detected. two-way talk stops thief
- **theft_QKocOIxfqu8** (taken): the package being left or taken was not detected. stranger with a walker
- **theft_r1cmcLUahKk** (taken): the package being left or taken was not detected. stranger stopped while owner away

## Known limitations

- The object detector (YOLOX-S, COCO classes) has no `box`/`package` class, so left/taken relies on comparing the doorstep before and after a visit; small parcels and night footage are the weak cases.
- A person and a parked car can both appear in a frame; the label priority picks the person.
- Alert volume: edited clips with many cuts raise many short person events; a real camera feed does not cut.
- Decider 2B on a laptop CPU takes about 10 s per decision when idle; the rule stub decides when it is off.
