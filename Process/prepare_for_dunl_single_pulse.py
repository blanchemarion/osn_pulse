import numpy as np
import pickle

# Load the dataset
with open("Data/animals_data_processed.pkl", "rb") as f:
    data = pickle.load(f)

valve_dict= data["valve_dict"]
phase_peaks_dict = data["phase_peaks_dict"]
#phase_hilbert_dict = data["phase_hilbert_dict"]
calcium_dict= data["calcium_dict"]
convolved_stim_dict= data["convolved_stim_dict"]

window_size = 50

all_data = []
for animal in valve_dict.keys():
    valve_data = valve_dict[animal]
    calcium_data = calcium_dict[animal]
    phase_peaks_data = phase_peaks_dict[animal]
    convolved_data = convolved_stim_dict[animal]

    valve_ts = np.arange(0, len(valve_data) / 1000, 0.001)  # 1 kHz sampling
    ca_ts = np.arange(0, len(calcium_data) / 10, 0.1)       # 10 Hz sampling
    
    #max_ca_duration = ca_ts[-1] # duration of calcium data
    
    #max_valve_idx = np.searchsorted(valve_ts, max_ca_duration) # index in valve_ts that corresponds to max ca duration

    #valve_dict[animal] = valve_data[:max_valve_idx]/100
    #phase_peaks_dict[animal] = phase_peaks_data[:max_valve_idx]
    calcium_dict[animal] = calcium_data.mean(axis=1).to_numpy()

    new_valve_ts = np.arange(0, len(valve_dict[animal]) / 1000, 0.001)
    # Detect events indices
    whiff_onsets = np.where(np.diff(valve_dict[animal]) > 0)[0]

    # Find phase and ca values around events
    phases = []
    calcium_resp = []
    onset_resp = []
    stim_values=[]
    for i in range(len(whiff_onsets)):
        start_idx = whiff_onsets[i]
        phase_value = np.mean(phase_peaks_dict[animal][start_idx:start_idx+50])  # Mean phase in the 50 ms following pulse onset (50 samples at 1kHz)
        phases.append(phase_value)

        # index of the value  of resp_ts that is the closest to stim_ts[start_idx]
        index = np.absolute(ca_ts-new_valve_ts[start_idx]).argmin()
        resp = calcium_dict[animal][index:index + 40] # calcium response in the 4s following the pulse onset (10Hz sampling)
        calcium_resp.append(resp)
        onset_resp.append(index)

        window_start = max(0, start_idx - window_size // 2)
        window_end = min(len(convolved_data), start_idx + window_size // 2)
        max_value = np.max(convolved_data[window_start:window_end])
        stim_values.append(max_value)

    data = {
        'whiff_onset': onset_resp,
        'phase': phases,
        'calcium': calcium_resp,
        'downsampled_convolved': stim_values
        }
    all_data.append(data)

animals = list(valve_dict.keys())
final_dict = dict(zip(animals, all_data))

with open("Data/ca_resp_to_pulse_data.pkl","wb") as f:
    pickle.dump(final_dict,f)
