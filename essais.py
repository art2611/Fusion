import numpy as np

#
# gall_feat_pool = np.zeros((2, 5))
# gall_feat_pool[0, :] = [0.2,0.3, 0.7, 0.9, 0.6]
# gall_feat_pool[1, :] = [4,2,3,1,5]
#
# print(-gall_feat_pool)
#
# print(np.argsort(-gall_feat_pool, axis = 1))

List = [0,0,0,0,1,2,3]
nw = [6,6,4,4,4,4,4]


List = np.array(List)
nw = np.array(nw)
print(List[nw])