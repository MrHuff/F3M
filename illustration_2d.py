import numpy as np
from matplotlib import pyplot as plt

a = np.random.rand(*(10000,2))
b = np.random.randn(*(10000,2))*0.25+0.5

plt.scatter(a[:,0],a[:,1],c='r',marker='.')
plt.scatter(b[:,0],b[:,1],c='b',marker='.')
plt.savefig('init.png',bbox_inches = 'tight')
plt.clf()

plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    labelleft=False,
    labelbottom=False,
    labeltop=False) #
plt.scatter(a[:,0],a[:,1],c='r',marker='.')
# plt.axis('off')
plt.savefig('a.png',bbox_inches = 'tight')
plt.clf()
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    labelleft=False,
    labelbottom=False,
    labeltop=False) #
plt.scatter(b[:,0],b[:,1],c='b',marker='.')
# plt.axis('off')
plt.savefig('b.png',bbox_inches = 'tight')
plt.clf()



