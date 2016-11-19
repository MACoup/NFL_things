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


fit1 = EllipticEnvelope(contamination=0.25).fit(X)

fit2 = EllipticEnvelope(contamination=0.25).fit(x, y)


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


def make_subplot_again(X, c, ax,pcX=0,pcY=1,fontSize=24,fontName='sans serif',ms=20,leg=True,title=None):
    outliers_fraction = 0.30
    clf = EllipticEnvelope(contamination=outliers_fraction)

    x = X['DK salary'].values
    y = X['points_per_dollar'].values.reshape(-1, 1)
    Xn = X
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
    Xn['pred'] = y_pred

# plot the levels lines and the points
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
    a = ax.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='burlywood')
    ax.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
    ax.axis('tight')

    # cvNames = np.sort(np.unique(covs[covariate]))
    # lines = []
    # for _i,i in enumerate(cvNames):
    #     indices = np.where(covs[covariate]==i)[0]
    #     s = ax.scatter(X[indices,pcX],X[indices,pcY],c=colors[_i],s=ms,label=covariate,alpha=0.9)
    #     lines.append(s)
    care_about = Xn[Xn['points_per_dollar'] > 3.5]
    care_about_false = care_about[care_about['pred'] == False]

    x_c_f = care_about_false['DK salary']
    y_c_f = care_about_false['points_per_dollar']
    ax.scatter(x_c_f, y_c_f, alpha=0.5, lw=2, edgecolor='k', s=50, marker='d', c='#5DC541', label='Great Value')

    dont_care_about = Xn[Xn['points_per_dollar'] <= 3.5]
    dont_care_about_false = dont_care_about[dont_care_about['pred'] == False]

    x_d_f = dont_care_about_false['DK salary']
    y_d_f = dont_care_about_false['points_per_dollar']
    ax.scatter(x_d_f, y_d_f, alpha=0.5, lw=2, s=70, marker='+', c='#6F0D73', label='Bad Value')

    Xn_true = Xn[Xn['pred'] == True]
    x_true = Xn_true['DK salary']
    y_true = Xn_true['points_per_dollar']
    ax.scatter(x_true, y_true, alpha=0.5, marker='o', c='#BD4864', label='Normal Value')

    ax.annotate('Ben\nRoethlisburger\nWeek 8 2014', fontsize=20, xy=(5800, 8.237931), xytext=(7300, 7), arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('Tom Brady\nWeek 11 2014', fontsize=20, xy=(9800, 1.640816), xytext=(7000, -0.5), arrowprops=dict(facecolor='black', shrink=0.05))


    ## axes
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize-2)

    ax.set_xlabel('Salary',fontsize=fontSize,fontname=fontName)
    ax.set_ylabel('Points per $1000',fontsize=fontSize,fontname=fontName)
    plt.locator_params(axis='x',nbins=5)
    ax.set_aspect(1./ax.get_data_ratio())
    ax.set_xlim(3000, 10000)
    ax.set_ylim(mm[1])
    ax.axhline(3.5, c='r', label='Threshold')
    box = ax.get_position()
    ax.set_position([box.x0 + box.width * 0.2, box.y0, box.width, box.height])
    ax.legend(loc='center right', bbox_to_anchor=(-0.2, 0.4), fontsize=20, scatterpoints=3, frameon=True)


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
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    make_subplot_again(X, c_x, ax1, leg=False, title='Quarterback Points')
    plt.axes().set_aspect(1/ax1.get_data_ratio())
    plt.show()
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
    outlier = df[df['points_per_dollar'] > 8]
    bad_outlier = df[(df['points_per_dollar'] < 2) & (df['DK salary'] > 8000)]
