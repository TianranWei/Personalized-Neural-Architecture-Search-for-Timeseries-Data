'''
Ranker variable changes the ranker:
1:  pointwise SVR
2: pointwise NN
3:  pairwise RankNet
'''
RANKER = 2

'''
Pointwise config:
'''
POINTWISE_EPSILON = 0.2
POINTWISE_KERNEL='linear'
POINTWISE_C=1

'''
Pairwise config:
'''
PAIRWISE_EPOCHS = 1000
PAIRWISE_BATCH_SIZE = 10
