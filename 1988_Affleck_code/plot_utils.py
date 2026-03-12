import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import matplotlib.colors as mcolors
from numpy import exp, angle
from config import N_MIN, N_MAX, T_MIN, T_MAX, SITE_N, N_NUM, T_NUM
import os
import time



def plot_phase_diagram(phases, timestamp, folder="results/figs", prefix="phase_diagram"):

    os.makedirs(folder, exist_ok=True)
    filename = f"{prefix}_{timestamp}.pdf"
    path = os.path.join(folder, filename)

    
    labels = ["Uniform", "Flux", "Kite", "Peierls", "Stripy", "Other"]
    cmap = ListedColormap(["gainsboro", "tomato", "deepskyblue", "gold", "skyblue", "black"])
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_array = np.vectorize(label_to_idx.get)(phases)

    nu_vals = np.linspace(N_MIN / SITE_N, N_MAX / SITE_N, N_NUM)
    t_vals = np.linspace(T_MIN, T_MAX, T_NUM)
    X, Y = np.meshgrid(nu_vals, t_vals)

    plt.figure(figsize=(4, 4))
    pc = plt.pcolormesh(X, Y, idx_array, cmap=cmap, shading="auto", vmin=-0.5, vmax=len(labels)-0.5, rasterized=True)
    plt.xticks([0.2, 0.3, 0.4, 0.5])
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.xlabel(r'$\nu$', fontsize=15)
    plt.ylabel(r'$t/J$', fontsize=15)

    cbar = plt.colorbar(pc, ticks=np.arange(len(labels)), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(labels)

    plt.title("Phase Diagram")
    plt.tight_layout()
    plt.gca().set_box_aspect(1)
    plt.savefig(path, dpi=600)
    print(f"Figure saved to {path}")
    plt.show()


def plot_order_parameters(data):
    chi1 = np.array([result[2] for result in data])
    chi2 = np.array([result[3] for result in data])
    chi3 = np.array([result[4] for result in data])
    chi4 = np.array([result[5] for result in data])
    phi1 = np.array([result[6] for result in data])
    phi2 = np.array([result[7] for result in data])
    phi3 = np.array([result[8] for result in data])
    phi4 = np.array([result[9] for result in data])
    E = np.array([result[10] for result in data])

    N=np.linspace(N_MIN, N_MAX,N_NUM)  
    t=np.linspace(T_MIN,T_MAX,T_NUM)    
    N_mesh, t_mesh = np.meshgrid(N, t)

    chi1 = chi1.reshape(N_mesh.shape)
    chi2 = chi2.reshape(N_mesh.shape)
    chi3 = chi3.reshape(N_mesh.shape)
    chi4 = chi4.reshape(N_mesh.shape)
    phi1 = phi1.reshape(N_mesh.shape)
    phi2 = phi2.reshape(N_mesh.shape)
    phi3 = phi3.reshape(N_mesh.shape)
    phi4 = phi4.reshape(N_mesh.shape)
    E  =  E.reshape(N_mesh.shape)
    '''
    chi1[0,:]=0
    chi2[0,:]=0
    chi3[0,:]=0
    chi4[0,:]=0
    phi1[0,:]=0
    phi2[0,:]=0
    phi3[0,:]=0
    phi4[0,:]=0
    '''
    phi1[np.abs(chi1)<0.001]=0
    phi2[np.abs(chi2)<0.001]=0
    phi3[np.abs(chi3)<0.001]=0
    phi4[np.abs(chi4)<0.001]=0

    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Times New Roman"

    plt.figure(figsize=(10,5))  
    vmax=1
    vmin=-1   
    cmap='RdBu' 
    
    plt.subplot(2,4,1)
    
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/SITE_N,t_mesh,chi1, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_1$")
    plt.xlabel('$\\nu$', fontsize=15) 
    plt.ylabel('$t/J$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\chi_1$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)
    
    
    
    plt.subplot(2,4,2)
    
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/SITE_N,t_mesh,chi2, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_2$")
    #plt.xlabel('$\\nu$', fontsize=15) 
    #plt.ylabel('$V$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\chi_2$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)
    
    
    plt.subplot(2,4,3)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/SITE_N,t_mesh,chi3, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_3$")
    #plt.xlabel('$\\nu$', fontsize=15) 
    #plt.ylabel('$V$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04) 
    clb.ax.set_ylabel(r'$\chi_3$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)
    
    
    plt.subplot(2,4,4)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/SITE_N,t_mesh,chi4, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_4$")
    #plt.xlabel('$\\nu$', fontsize=15) 
    #plt.ylabel('$V$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\chi_4$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)
    

    vmax=np.pi
    vmin=-np.pi
    plt.subplot(2,4,5) 
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/SITE_N,t_mesh,phi1, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_4$")
    plt.xlabel('$\\nu$', fontsize=15) 
    plt.ylabel('$t/J$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\phi_1$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)


    plt.subplot(2,4,6)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/SITE_N,t_mesh,phi2, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_3$")
    plt.xlabel('$\\nu$', fontsize=15) 
    #plt.ylabel('$V$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04) 
    clb.ax.set_ylabel(r'$\phi_2$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)
    
    
    plt.subplot(2,4,7)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/SITE_N,t_mesh,phi3, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_4$")
    plt.xlabel('$\\nu$', fontsize=15) 
    #plt.ylabel('$V$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\phi_3$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)


    plt.subplot(2,4,8)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/SITE_N,t_mesh,phi4, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_4$")
    plt.xlabel('$\\nu$', fontsize=15) 
    #plt.ylabel('$V$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\phi_4$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)
    

    plt.tight_layout()
    plt.show()



def plot_flux_square(data, N_MIN, N_MAX, T_MIN, T_MAX, SITE_N, N_NUM, T_NUM):
    chi1 = np.array([row[2] for row in data])
    chi2 = np.array([row[3] for row in data])
    chi3 = np.array([row[4] for row in data])
    chi4 = np.array([row[5] for row in data])
    phi1 = np.array([row[6] for row in data])
    phi2 = np.array([row[7] for row in data])
    phi3 = np.array([row[8] for row in data])
    phi4 = np.array([row[9] for row in data])

    N = np.linspace(N_MIN, N_MAX, N_NUM)
    t = np.linspace(T_MIN, T_MAX, T_NUM)
    N_mesh, t_mesh = np.meshgrid(N, t)

    chi1 = chi1.reshape(N_mesh.shape)
    chi2 = chi2.reshape(N_mesh.shape)
    chi3 = chi3.reshape(N_mesh.shape)
    chi4 = chi4.reshape(N_mesh.shape)
    phi1 = phi1.reshape(N_mesh.shape)
    phi2 = phi2.reshape(N_mesh.shape)
    phi3 = phi3.reshape(N_mesh.shape)
    phi4 = phi4.reshape(N_mesh.shape)

    theta1 = angle(t_mesh + chi1 * exp(-1j * phi1))
    theta2 = angle(t_mesh + chi2 * exp(-1j * phi2))
    theta3 = angle(t_mesh + chi3 * exp(-1j * phi3))
    theta4 = angle(t_mesh + chi4 * exp(-1j * phi4))
    flux = np.abs(-theta1 + theta2 - theta3 + theta4)

    plt.figure()
    plt.pcolor(N_mesh / SITE_N, t_mesh, flux, shading='auto', cmap='Blues', rasterized=True)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\Phi$', rotation=270, labelpad=10, fontsize=15)
    plt.xlabel('$\nu$', fontsize=15)
    plt.ylabel('$t/J$', fontsize=15)
    plt.gca().tick_params(axis='both', direction='in')
    plt.gca().set_box_aspect(1)
    plt.tight_layout()
    plt.show()
