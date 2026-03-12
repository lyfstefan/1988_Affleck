import numpy as np
import time
from solver import get_phasediagram_data
from data_utils import save_data, load_data
from phase_classify import classify_phases, filter_phis_by_chis
from plot_utils import plot_order_parameters, plot_phase_diagram
import importlib
import config
importlib.reload(config)

if __name__ == "__main__":
    
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), 'Start')

    data = get_phasediagram_data()
    save_data(data, timestamp=timestamp)

    filtered_data = filter_phis_by_chis(data)
    plot_order_parameters(filtered_data)

    phases = classify_phases(filtered_data, chi_tol=5e-2, phi_tol=2e-1, peierls_ratio=1.1)
    plot_phase_diagram(phases, timestamp=timestamp)
    
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), 'End')
