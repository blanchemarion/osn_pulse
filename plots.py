import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="sparseness/Data/animals_data_processed.pkl",
    )
    args = parser.parse_args()
    params = vars(args)

    return params


def normalize_signal(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

def plot_signals_normalized(data, animal, t_start=None, t_end=None):

    valve = data["valve_dict"][animal] / 100  
    #breath = data["breath_filt_dict"][animal]
    calcium_signal = data["ca_interp_dict"][animal].mean(axis=1)
    
    t = np.arange(len(valve))
    
    if t_start is not None or t_end is not None:
        if t_start is None:
            t_start = 0
        if t_end is None:
            t_end = len(t)
        t = t[t_start:t_end]
        valve = valve[t_start:t_end]
        #breath = breath[t_start:t_end]
        calcium_signal = calcium_signal[t_start:t_end]
    
    valve_norm = normalize_signal(valve)
    #breath_norm = normalize_signal(breath)
    calcium_norm = normalize_signal(calcium_signal)
    
    plt.figure(figsize=(12, 4))
    plt.plot(t, valve_norm, label="Valve (normalized)", color="red", linewidth=1, alpha=0.8)
    #plt.plot(t, breath_norm, label="Breath (normalized)", color="green")
    plt.plot(t, calcium_norm, label="Calcium (normalized)", color="green", linewidth=1)
    
    plt.xticks(np.linspace(t[0], t[-1], num=10), np.linspace(t[0]/1000, t[-1]/1000, num=10).round(2))
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(animal)
    #plt.legend()
    plt.show()

def plot_signals_all(data, animal, t_start=None, t_end=None):
    valve = data["valve_dict"][animal] / 100
    ca_traces = data["ca_interp_dict"][animal]  # shape: (time, n_rois)

    t = np.arange(len(valve))

    # Subset time if needed
    if t_start is not None or t_end is not None:
        if t_start is None:
            t_start = 0
        if t_end is None:
            t_end = len(t)
        t = t[t_start:t_end]
        valve = valve[t_start:t_end]
        ca_traces = ca_traces.values[t_start:t_end, :]

    valve_norm = normalize_signal(valve)

    plt.figure(figsize=(12, 4))

    # Plot normalized calcium traces for each ROI
    for i in range(ca_traces.shape[1]):
        trace = ca_traces[:, i]
        trace_norm = normalize_signal(trace)
        plt.plot(t, trace_norm, alpha=0.6, linewidth=0.8, label=f"ROI {i}" if i < 10 else None)  # label only first 10

    # Plot valve signal on top
    plt.plot(t, valve_norm, label="Valve (normalized)", color="red", linewidth=1)

    plt.xticks(np.linspace(t[0], t[-1], num=10), np.linspace(t[0]/1000, t[-1]/1000, num=10).round(2))
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def main():

    params = init_params()
    
    animal = 'HW1'

    with open(params["data_path"], "rb") as f:
        data = pickle.load(f)
        
    plot_signals_all(data, animal, t_start=100000, t_end=125000)

    plot_signals_normalized(data, animal, t_start=100000, t_end=125000)
        

if __name__ == "__main__":
    main()
