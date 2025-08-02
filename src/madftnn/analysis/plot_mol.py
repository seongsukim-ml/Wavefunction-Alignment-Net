import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from .build_chem_graph import build_graph

def get_cmap_coloar(x,lut = 50,x_min = None,x_max = None):
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    x = np.clip(x,x_min,x_max)
    x = 1.0*(x-x_min)/(x_max-x_min)*lut
    color = [plt.get_cmap("Reds",lut)(int(i)) for i in x]
    plt.set_cmap(plt.get_cmap("Reds",lut))

    return color

def ax_visualize(pos,atomic_numbers,atom_labels=None,force_error = None,x_min = 0,x_max = 0.05):
    # creating figure
    fig = plt.figure()
    ax = Axes3D(fig)
    if force_error is not None:
        color = get_cmap_coloar(force_error,10,x_min,x_max)
    elif atom_labels is not None:
        color = atom_labels #line.labels.detach().cpu().numpy()
        plt.set_cmap(plt.get_cmap("Paired",40))
    else:
        color = atomic_numbers
        plt.set_cmap(plt.get_cmap("Paired",40))

    im = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=color)

    # creating the plot
    # im = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=atomic_numbers)
    G = build_graph(atomic_numbers,pos)
    for edge in G.edges:
        src,tgt = edge
        ax.plot3D([pos[src,0],pos[tgt,0]], [pos[src,1],pos[tgt,1]], [pos[src,2],pos[tgt,2]],'gray')    #绘制空间曲线

    # setting title and labels
    ax.set_title("3D plot")
    ax.set_xlabel('x-axis:')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    if force_error is not None:
        plt.colorbar(im,format = matplotlib.ticker.FuncFormatter(lambda x,pos:x*(x_max-x_min)+x_min))
    # displaying the plot
    plt.show()