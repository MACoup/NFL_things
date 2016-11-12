import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.covariance import EllipticEnvelope
import sys
sys.path.append('/Users/MACDaddy/fantasy_football/NFL_things/nfldb_queries/')
from pandas_nfldb_dfs import passing, rec, rush, te, dst
from Final_DF import FinalDF
from scipy.stats import scoreatpercentile
from sklearn import preprocessing

fin = FinalDF(season_type='Regular', position='QB')
df = fin.get_df()
df = df.replace([np.inf, -np.inf], 0)


X = df[['DK salary', 'points_per_dollar']]
x = df['DK salary'].values.reshape(-1, 1)
y = df['points_per_dollar'].values.reshape(-1, 1)
c_x = df['DK points']

X_scaled = preprocessing.scale(X)
x_scaled = preprocessing.scale(x)
y_scaled = preprocessing.scale(y)

fit1 = EllipticEnvelope(contamination=0.25).fit(X)

fit2 = EllipticEnvelope(contamination=0.25).fit(x, y)

fit3 = EllipticEnvelope(contamination=0.25).fit(X_scaled)

fit4 = EllipticEnvelope(contamination=0.25).fit(x_scaled, y_scaled)

def make_subplot(X, covs, ax, covariate=None, pcX=0,pcY=1,fontSize=10,fontName='sans serif',ms=20,leg=True,title=None):
    colors = ['k','cyan','r','orange','g','b','magenta']
    cvNames = np.sort(np.unique(covs[covariate]))
    lines = []


    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize-2)

    buff = 0.02
    bufferX = buff * (fit[:,pcX].max() - fit[:,pcX].min())
    bufferY = buff * (fit[:,pcY].max() - fit[:,pcY].min())
    ax.set_xlim([fit[:,pcX].min()-bufferX,fit[:,pcX].max()+bufferX])
    ax.set_ylim([fit[:,pcY].min()-bufferY,fit[:,pcY].max()+bufferY])
    ax.set_xlabel("D-%s"%str(pcX+1),fontsize=fontSize,fontname=fontName)
    ax.set_ylabel("D-%s"%str(pcY+1),fontsize=fontSize,fontname=fontName)
    plt.locator_params(axis='x',nbins=5)
    ax.set_aspect(1./ax.get_data_ratio())

    if title:
        ax.set_title(title,fontsize=fontSize+2,fontname=fontName)
    if leg:
        legend = ax.legend(lines,cvNames,loc='upper right',scatterpoints=1,
                           handletextpad=0.01,labelspacing=0.01,borderpad=0.1,handlelength=1.0)

        for label in legend.get_texts():
            label.set_fontsize(fontSize-2)
            label.set_fontname(fontName)


def make_subplot_again(X, c, ax,pcX=0,pcY=1,fontSize=10,fontName='sans serif',ms=20,leg=True,title=None):
    outliers_fraction = 0.30
    clf = EllipticEnvelope(contamination=outliers_fraction)

    x = X['DK salary'].values
    y = X['points_per_dollar'].values.reshape(-1, 1)

    X = X.values

    buff = 0.02
    bufferX = buff * (X[:,pcX].max() - X[:,pcX].min())
    bufferY = buff * (X[:,pcY].max() - X[:,pcY].min())
    mm = [(X[:,pcX].min()-bufferX,X[:,pcX].max()+bufferX),(X[:,pcY].min()-bufferY,X[:,pcY].max()+bufferY)]
    xx, yy = np.meshgrid(np.linspace(mm[0][0],mm[0][1], 500), np.linspace(mm[1][0],mm[1][1],500))

    # fit the data and tag outliers

    clf.fit(X)
    y_pred = clf.decision_function(X).ravel()
    threshold = scoreatpercentile(y_pred,100 * outliers_fraction)

    y_pred = y_pred > threshold

    print y_pred

# plot the levels lines and the points
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape) 

    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
    a = ax.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
    ax.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
    ax.axis('tight')

    # cvNames = np.sort(np.unique(covs[covariate]))
    # lines = []
    # for _i,i in enumerate(cvNames):
    #     indices = np.where(covs[covariate]==i)[0]
    #     s = ax.scatter(X[indices,pcX],X[indices,pcY],c=colors[_i],s=ms,label=covariate,alpha=0.9)
    #     lines.append(s)
    ax.scatter(x, y, alpha=0.5, c=c, cmap='afmhot_r')


    ## axes
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize-2)

    ax.set_xlabel('DK salary',fontsize=fontSize,fontname=fontName)
    ax.set_ylabel('points_per_dollar',fontsize=fontSize,fontname=fontName)
    plt.locator_params(axis='x',nbins=5)
    ax.set_aspect(1./ax.get_data_ratio())
    ax.set_xlim(mm[0])
    ax.set_ylim(mm[1])

    if title:
        ax.set_title(title,fontsize=fontSize+2,fontname=fontName)
    # if leg:
    #     legend = ax.legend(lines,cvNames,loc='upper right',scatterpoints=1,
    #                        handletextpad=0.01,labelspacing=0.01,borderpad=0.1,handlelength=1.0)

        # for label in legend.get_texts():
        #     label.set_fontsize(fontSize-2)
        #     label.set_fontname(fontName)

## variables
if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    make_subplot_again(X, c_x, ax1, leg=False, title='Quarterback')
    # make_subplot_again(X_scaled, ax2)
#
# ## make the figure again
# plt.clf()
# fig = plt.figure()
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)
# make_subplot_again(fit1,covs,'phenotype',ax1,pcX=0,pcY=1,leg=True,title='PCA-raw')
# make_subplot_again(fit2,covs,'phenotype',ax2,pcX=0,pcY=1,leg=False,title='tSNE-raw')
# make_subplot_again(fit3,covs,'phenotype',ax3,pcX=0,pcY=1,leg=False,title='PCA-scaled')
# make_subplot_again(fit4,covs,'phenotype',ax4,pcX=0,pcY=1,leg=False,title='tSNE-scaled')
# ax1.set_xlabel("")
# ax2.set_xlabel("")
# ax2.set_ylabel("")
# ax4.set_ylabel("")
# plt.subplots_adjust(hspace=0.3,wspace=0.05)
# plt.savefig("outliers-detection.png",dpi=600)
