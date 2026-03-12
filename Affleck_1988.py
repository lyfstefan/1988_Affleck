
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.colors as mcolors
from numpy import pi, sin, cos, exp, sqrt, tan
import matplotlib.colors as mcolors
from scipy.optimize import minimize
from scipy.optimize import shgo, basinhopping, dual_annealing, approx_fprime
from scipy.optimize import brentq
from scipy.optimize import bisect
import scipy.integrate as spi
import time   
from joblib import Parallel, delayed        



def get_E12_and_N(chi1, chi2, chi3, chi4, phi1, phi2, phi3, phi4, t):
    h12 = - (t + chi1 * np.exp(-1j*phi1)) * np.exp(1j * Kx) \
          - (t + chi2 * np.exp(-1j*phi2)) * np.exp(1j * Ky) \
          - (t + chi3 * np.exp(-1j*phi3)) * np.exp(-1j * Kx) \
          - (t + chi4 * np.exp(-1j*phi4)) * np.exp(-1j * Ky)

    common_term = -4*tp*np.cos(Kx)*np.cos(Ky)
    abs_h12 = np.abs(h12)**0.5
    E1_matrix = common_term - abs_h12
    E2_matrix = common_term + abs_h12

    return E1_matrix, E2_matrix


def Mu12_from_E(E1, E2, N_target):
    start = -1000.0
    end = 1000.0
    max_iter = 100

    for _ in range(max_iter):
        mid = 0.5 * (start + end)

        N_mid = ((E1 <= mid).sum() + (E2 <= mid).sum()) / 2

        if abs(N_mid - N_target) <= error / 2:
            return mid
        elif N_mid > N_target:
            end = mid
        else:
            start = mid

    return mid  # fallback if not converged




def Etot(x, N, t):
    chi1, chi2, chi3, chi4, phi1, phi2, phi3, phi4 = x
    E1_matrix, E2_matrix = get_E12_and_N(chi1, chi2, chi3, chi4, phi1, phi2, phi3, phi4, t)
    mu = Mu12_from_E(E1_matrix, E2_matrix, N)

    E1_matrix[E1_matrix > mu] = 0
    E2_matrix[E2_matrix > mu] = 0

    E = (E1_matrix.sum() + E2_matrix.sum())/2 + siteN/(2*J)*(chi1**2 + chi2**2 + chi3**2 + chi4**2)
    return E / siteN

'''
def Order(N,t):

    bounds = [(0, 1),(0, 1),(0, 1),(0, 1),(-pi, pi),(-pi, pi),(-pi, pi),(-pi, pi)]  
    #result = shgo(Etot,args=(N,t),bounds=bounds,sampling_method='sobol',n=256,iters=1)
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "args": (N, t)}
    result = basinhopping(Etot, x0=np.random.rand(8), minimizer_kwargs=minimizer_kwargs, niter=200)

    return N, t, result.x[0],result.x[1],result.x[2],result.x[3],\
        result.x[4],result.x[5],result.x[6],result.x[7],result.fun


'''



def Order(N, t):
    
    bounds = [(0, 1)] * 4 + [(-np.pi, np.pi)] * 4

    # Step 1: 粗略 dual annealing，只跑少量迭代
    result = dual_annealing(
        Etot,
        bounds=bounds,
        args=(N, t),
        maxiter=512,              # 快速收敛
        initial_temp=500.0,
        restart_temp_ratio=1e-5,
        visit=2.6,
        accept=-4.0,
        no_local_search=True,    # 不局部优化，提高速度
        seed=None
    )

    # Step 2: 精化，单次 L-BFGS-B 提高局部精度（快）
    
    local_result = minimize(
        Etot,
        result.x,
        args=(N, t),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 2000
        }
    )
    '''
    local_result = minimize(
        Etot,
        result.x,
        args=(N, t),
        method="TNC",
        bounds=bounds,
        options={
            "maxfun": 5000
        }
    )
    '''
    x = local_result.x
    return (N, t, *x, local_result.fun)

def Order_multi(N, t):
    bounds = [(0, 1)] * 4 + [(-np.pi, np.pi)] * 4
    best_result = None
    best_fun = np.inf
    if t >= 0.2 and N/siteN <= 0.45:
        return Order(N, t)  # 对于小t直接调用单点优化
    else:
        for i in range(8):
            # Step 1: 粗略全局搜索（不同种子探索不同区域）
            result = dual_annealing(
                Etot,
                bounds=bounds,
                args=(N, t),
                maxiter=512,
                initial_temp=500.0,
                restart_temp_ratio=1e-5,
                visit=2.6,
                accept=-4.0,
                no_local_search=True,
                seed=i  # 多起点依赖 seed 不同
            )

            # Step 2: 精细局部优化
            local_result = minimize(
                Etot,
                result.x,
                args=(N, t),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": 2000
                }
            )

            # Step 3: 记录最优结果
            if local_result.fun < best_fun:
                best_fun = local_result.fun
                best_result = local_result

        x = best_result.x
        return (N, t, *x, best_result.fun)


def get_phasediagram_data():
    N_list = np.linspace(Nmin, Nmax, N_num)
    t_list = np.linspace(tmin, tmax, t_num)
    task_list = [(N, t) for t in t_list for N in N_list]

    results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
        delayed(Order_multi)(N, t) for N, t in task_list
    )

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), 'Order parameters found!')
    return results

def filter_phis_by_chis(data, chi_threshold=1e-3):
    """
    Set phi_i to zero where corresponding chi_i is below threshold.

    Parameters:
    - data: 2D array-like, shape (n_samples, >=10)
        Each row should be [N, t, chi1, chi2, chi3, chi4, phi1, phi2, phi3, phi4, ...]
    - chi_threshold: float, threshold below which phi is set to zero

    Returns:
    - filtered_data: np.ndarray with modified phis
    """
    data = np.array(data, copy=True)  
    chis = data[:, 2:6]
    phis = data[:, 6:10]

    # filter phis
    mask = np.abs(chis) < chi_threshold
    phis[mask] = 0.0
    data[:, 6:10] = phis

    return data


def classify_phases(filtered_data, chi_tol=1e-2, phi_tol=1e-2, peierls_ratio=1.1):
    phase_names = []

    for row in filtered_data:
        t = row[1]
        chi1, chi2, chi3, chi4 = row[2:6]
        phi1, phi2, phi3, phi4 = row[6:10]
        
        chis = np.array([chi1, chi2, chi3, chi4])
        phis = np.array([phi1, phi2, phi3, phi4])
        theta1 = np.angle( t + chi1 * exp(-1j*phi1))
        theta2 = np.angle( t + chi2 * exp(-1j*phi2))
        theta3 = np.angle( t + chi3 * exp(-1j*phi3))
        theta4 = np.angle( t + chi4 * exp(-1j*phi4))
        flux=-theta1+theta2-theta3+theta4
        flux=np.abs(flux)

        # Condition 1: Uniform
        if np.std(chis) < chi_tol and (np.all(np.abs(phis) < phi_tol) or flux/np.pi%2<1e-2):
            phase_names.append("Uniform")
            continue

        # Condition 2: Flux (phi alternates in sign, chi almost equal)
        phi_signs = np.sign(np.real(phis))
        if np.std(chis) < chi_tol and (
            np.all(phi_signs == [+1, -1, +1, -1]) or np.all(phi_signs == [-1, +1, -1, +1])
        ):
            phase_names.append("Flux")
            continue

        # Condition 3: Kite (phi ≈ 0, chi grouped 2 by 2)
        
        chi_patterns = [
            (chis[0], chis[1], chis[2], chis[3]),  # chi1=chi2 ≠ chi3=chi4
            (chis[1], chis[2], chis[3], chis[0]),  # chi2=chi3 ≠ chi4=chi1
            (chis[2], chis[3], chis[0], chis[1]),  # chi3=chi4 ≠ chi1=chi2
            (chis[3], chis[0], chis[1], chis[2]),  # chi4=chi1 ≠ chi2=chi3
        ]
        matched_kite = False
        for ch in chi_patterns:
            if abs(ch[0] - ch[1]) < chi_tol and abs(ch[2] - ch[3]) < chi_tol and abs(ch[0] - ch[2]) > chi_tol:
                phase_names.append("Kite")
                matched_kite = True
                break
        if matched_kite:
                continue

        # Condition 4: Peierls (one chi significantly larger)
        chi_abs = np.abs(chis)
        max_val = np.max(chi_abs)
        rest = np.delete(chi_abs, np.argmax(chi_abs))
        if max_val > peierls_ratio * np.max(rest):
            phase_names.append("Peierls")
            continue

        # Condition 5: Stripy (chi1=chi3≠chi2=chi4)
        if abs(chis[0] - chis[2]) < chi_tol and abs(chis[1] - chis[3]) < chi_tol and abs(chis[0] - chis[3]) > chi_tol:
            phase_names.append("Stripy")
            continue

        # Fallback: Other
        phase_names.append("Other")

    return np.array(phase_names).reshape((t_num, N_num))


def plot_phase_diagram(phases):
    """
    Plot a phase diagram using pcolormesh from a 2D array of phase names.

    Parameters:
    - phases: 2D np.array of strings (shape = [t_num, nu_num])
    - filename: name of saved figure (without extension)
    """
    # 设置颜色和标签
    labels = ["Uniform", "Flux", "Kite", "Peierls", "Other", "Stripy"]
    cmap = ListedColormap(["gainsboro", "tomato", "deepskyblue", "gold", "green", "black"])
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # 转为整数 index 矩阵
    idx_array = np.vectorize(label_to_idx.get)(phases)

    # 构建网格坐标（assume you have access to these globals）
    nu_vals = np.linspace(Nmin / siteN, Nmax / siteN, N_num)
    t_vals  = np.linspace(tmin, tmax, t_num)
    X, Y = np.meshgrid(nu_vals, t_vals)

    # 画图
    plt.figure(figsize=(4, 4))
    pc = plt.pcolormesh(X, Y, idx_array, cmap=cmap, shading="auto", vmin=-0.5, vmax=len(labels)-0.5)

    # 设置 ticks 和标签
    x_tick_vals = [0.2, 0.3, 0.4, 0.5]
    y_tick_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    plt.xticks(x_tick_vals, [f"{v:.1f}" for v in x_tick_vals])
    plt.yticks(y_tick_vals, [f"{v:.1f}" for v in y_tick_vals])
    plt.xlabel(r'$\nu$', fontsize=15)
    plt.ylabel(r'$t/J$', fontsize=15)

    # 添加 colorbar
    cbar = plt.colorbar(pc, ticks=np.arange(len(labels)), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(labels)

    plt.title("Phase Diagram")
    plt.tight_layout()
    plt.gca().set_box_aspect(1)
    plt.savefig(f"{filename}.pdf", dpi=300)
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

    N=np.linspace(Nmin, Nmax,N_num)  
    t=np.linspace(tmin,tmax,t_num)    
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
    plt.pcolor(N_mesh/siteN,t_mesh,chi1, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_1$")
    plt.xlabel('$\\nu$', fontsize=15) 
    plt.ylabel('$t/J$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\chi_1$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)
    
    
    
    plt.subplot(2,4,2)
    
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/siteN,t_mesh,chi2, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_2$")
    #plt.xlabel('$\\nu$', fontsize=15) 
    #plt.ylabel('$V$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\chi_2$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)
    
    
    plt.subplot(2,4,3)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/siteN,t_mesh,chi3, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_3$")
    #plt.xlabel('$\\nu$', fontsize=15) 
    #plt.ylabel('$V$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04) 
    clb.ax.set_ylabel(r'$\chi_3$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)
    
    
    plt.subplot(2,4,4)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/siteN,t_mesh,chi4, shading='auto',cmap=cmap,norm=norm,rasterized=True)
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
    plt.pcolor(N_mesh/siteN,t_mesh,phi1, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_4$")
    plt.xlabel('$\\nu$', fontsize=15) 
    plt.ylabel('$t/J$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\phi_1$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)


    plt.subplot(2,4,6)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/siteN,t_mesh,phi2, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_3$")
    plt.xlabel('$\\nu$', fontsize=15) 
    #plt.ylabel('$V$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04) 
    clb.ax.set_ylabel(r'$\phi_2$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)
    
    
    plt.subplot(2,4,7)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/siteN,t_mesh,phi3, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_4$")
    plt.xlabel('$\\nu$', fontsize=15) 
    #plt.ylabel('$V$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\phi_3$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)


    plt.subplot(2,4,8)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax = vmax, vcenter=0)
    plt.pcolor(N_mesh/siteN,t_mesh,phi4, shading='auto',cmap=cmap,norm=norm,rasterized=True)
    #plt.title( "Phase Diagram-$J_4$")
    plt.xlabel('$\\nu$', fontsize=15) 
    #plt.ylabel('$V$', fontsize=15)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.ax.set_ylabel(r'$\phi_4$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)
    

    plt.tight_layout()
    #plt.savefig(f"Order_parameters_{filename}.pdf", bbox_inches='tight')
    plt.show()


def plot_flux_square(data):

    chi1 = np.array([result[2] for result in data])
    chi2 = np.array([result[3] for result in data])
    chi3 = np.array([result[4] for result in data])
    chi4 = np.array([result[5] for result in data])
    phi1 = np.array([result[6] for result in data])
    phi2 = np.array([result[7] for result in data])
    phi3 = np.array([result[8] for result in data])
    phi4 = np.array([result[9] for result in data])
    E = np.array([result[10] for result in data])

    N=np.linspace(Nmin, Nmax,N_num)  
    t=np.linspace(tmin,tmax,t_num)    
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

    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Times New Roman"


    theta1 = np.angle( t_mesh + chi1 * exp(-1j*phi1))
    theta2 = np.angle( t_mesh + chi2 * exp(-1j*phi2))
    theta3 = np.angle( t_mesh + chi3 * exp(-1j*phi3))
    theta4 = np.angle( t_mesh + chi4 * exp(-1j*phi4))
    flux=-theta1+theta2-theta3+theta4
    flux=np.abs(flux)
    plt.figure()
    
    plt.pcolor(N_mesh/siteN,t_mesh,flux, shading='auto',cmap='Blues',rasterized=True)
    clb = plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel('$\\nu$', fontsize=15)
    plt.ylabel('$t/J$', fontsize=15)
    clb.ax.set_ylabel(r'$\Phi$', rotation=270,labelpad=10,fontsize=15)
    plt.gca().tick_params(axis='both' , direction='in')
    plt.gca().set_box_aspect(1)
    plt.show()



#Define parameters
J=1
tp=0          #next nearest hopping
Nx=50         #the number of sites on x-direction
Ny=50         #the number of sites on y-direction
siteN=Nx*Ny   #the total number of sites   
N_num=20
t_num=20
Nmin, Nmax = 0.2*siteN , 0.5*siteN 
tmin, tmax = 0, 0.5
error=3        #the difference between the actual number of particles and the expected number of particles, 
               #3 is good

kx = np.arange(0, 2*np.pi, 2*np.pi/Nx) - np.pi
ky = np.arange(0, 2*np.pi, 2*np.pi/Ny) - np.pi
Kx, Ky = np.meshgrid(kx, ky)


if __name__ == "__main__":        


    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'Start')
    
    data=get_phasediagram_data()


    filename = f"phasediagram_data_{time.strftime('%Y%m%d_%H%M%S')}.npy"
    np.save(filename, data)
    #data = np.load("filename", allow_pickle=True)
    filtered_data = filter_phis_by_chis(data)                
    plot_order_parameters(filtered_data)  # 绘制有序参量图                 
    phases = classify_phases(filtered_data, chi_tol=2e-2, phi_tol=2e-1, peierls_ratio=1.3)                   # 分类
    plot_phase_diagram(phases)                                 
    
    #plot_flux_square(data)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'End')

    