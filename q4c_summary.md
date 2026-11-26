clustering: kmeans k=10 (fit on train reps)  TEST ARI 0.0599  NMI 0.1028   | SVM test acc 50.99

| class | SVM recall | majority cluster | cluster recall | cluster precision |
|---|---|---|---|---|
| airplane | 0.545 | 1 | 0.323 | 0.455 |
| automobile | 0.639 | 3 | 0.189 | 0.285 |
| bird | 0.375 | 5 | 0.309 | 0.167 |
| cat | 0.390 | 9 | 0.160 | 0.160 |
| deer | 0.421 | 5 | 0.342 | 0.185 |
| dog | 0.395 | 2 | 0.250 | 0.210 |
| frog | 0.580 | 5 | 0.472 | 0.255 |
| horse | 0.514 | 0 | 0.227 | 0.231 |
| ship | 0.643 | 8 | 0.375 | 0.335 |
| truck | 0.597 | 0 | 0.244 | 0.248 |

| class pair | SVM pairwise acc | best acc from cluster id alone | gap | same majority cluster |
|---|---|---|---|---|
| automobile-horse | 0.904 | 0.643 | +0.261 | False |
| horse-truck | 0.875 | 0.645 | +0.229 | True |
| automobile-dog | 0.917 | 0.697 | +0.221 | False |
| automobile-deer | 0.924 | 0.712 | +0.213 | False |
| horse-ship | 0.922 | 0.721 | +0.201 | False |
| automobile-frog | 0.928 | 0.734 | +0.195 | False |
| automobile-bird | 0.900 | 0.706 | +0.195 | False |
| automobile-cat | 0.890 | 0.698 | +0.193 | False |
| deer-frog | 0.784 | 0.595 | +0.190 | True |
| cat-horse | 0.816 | 0.627 | +0.189 | False |

contingency (rows = true class, cols = cluster):

```
  60  323  182   37   27   83   47   22  156   63
 113   11   43  189   84  118  136   56  187   63
  42   90  149   24   60  309  131   83   41   71
  56   21  149   39  145  154  105  141   30  160
  55   28  114   34   37  342  157   44   35  154
  46   15  250   22  105  158  121  118   43  122
  18    7  101   49   64  472   92   61   23  113
 227   21   84   44   75  110  131   67   68  173
 122  154   86   42   65   35   41   47  375   33
 244   40   35  183   75   67   60   90  161   45
```
