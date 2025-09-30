import numpy as np
import pickle

# Load the dataset
with open("Data/animals_data_processed.pkl", "rb") as f:
    data = pickle.load(f)

valve_dict= data["valve_dict"]
phase_peaks_dict = data["phase_peaks_dict"]
calcium_dict= data["calcium_dict"]

all_data = []
for animal in valve_dict.keys():
    valve_data = valve_dict[animal]
    calcium_data = calcium_dict[animal]
    phase_peaks_data = phase_peaks_dict[animal]

    valve_ts = np.arange(0, len(valve_data) / 1000, 0.001)  # 1 kHz sampling
    ca_ts = np.arange(0, len(calcium_data) / 10, 0.1)       # 10 Hz sampling
    
    max_ca_duration = ca_ts[-1] # duration of calcium data
    
    max_valve_idx = np.searchsorted(valve_ts, max_ca_duration) # index in valve_ts that corresponds to max ca duration

    valve_dict[animal] = valve_data[:max_valve_idx]/100
    phase_peaks_dict[animal] = phase_peaks_data[:max_valve_idx]
    calcium_dict[animal] = calcium_data.mean(axis=1).to_numpy()

    new_valve_ts = np.arange(0, len(valve_dict[animal]) / 1000, 0.001)
    # Detect events indices
    whiff_onsets = np.where(np.diff(valve_dict[animal]) > 0)[0]

    phases1=[]
    phases2=[]
    calcium_resp=[]
    onset_pulse1=[]
    onset_pulse2=[]
    indices_pulse2=[]
    for i in range(1, len(whiff_onsets)):
        diff = whiff_onsets[i] - whiff_onsets[i - 1]
        if diff < 1500:
            start_idx_pulse1= whiff_onsets[i-1]
            pulse1_phase= np.mean(phase_peaks_dict[animal][start_idx_pulse1:start_idx_pulse1+75])  # Mean phase in the 75 ms following pulse onset (75 samples at 1kHz)
            phases1.append(pulse1_phase)

            start_idx_pulse2= whiff_onsets[i]
            pulse2_phase= np.mean(phase_peaks_dict[animal][start_idx_pulse2:start_idx_pulse2+75])  # Mean phase in the 75 ms following pulse onset (75 samples at 1kHz)
            phases2.append(pulse2_phase)

            # index of the value  of resp_ts (ca_ts) that is the closest to stim_ts[start_idx] (new_valve_ts[start_idx])
            index_pulse1 = np.absolute(ca_ts-new_valve_ts[start_idx_pulse1]).argmin()
            index_pulse2 = np.absolute(ca_ts-new_valve_ts[start_idx_pulse2]).argmin()

            resp = calcium_dict[animal][index_pulse1:index_pulse2 + 40] # calcium response start at pulse 1 onset and stops 3s after pulse 2 onset (10Hz sampling)
            calcium_resp.append(resp)

            # then in the ref frame of each trial: the index of whiff1 is 0 and the index of whiff2 is index of pulse2 - index of pulse 1 in the original signal
            onset_pulse1.append(0)
            onset_pulse2.append(index_pulse2 - index_pulse1)

            indices_pulse2.append(index_pulse2)

    data = {
        'whiff1_onset': onset_pulse1,
        'whiff2_onset':onset_pulse2,
        'whiff2_index': indices_pulse2,
        'phase1': phases1,
        'phase2': phases2,
        'calcium': calcium_resp
        }
    all_data.append(data)

animals = list(valve_dict.keys())
final_dict = dict(zip(animals, all_data))

with open("Data/ca_resp_to_paired_pulses_data.pkl","wb") as f:
    pickle.dump(final_dict,f)