import numpy as np
import pickle

# Load the dataset
with open("Data/animals_data_processed.pkl", "rb") as f:
    data = pickle.load(f)

convolved_stim = data["convolved_stim_dict"]
phase_peaks_dict = data["phase_peaks_dict"]
calcium_dict= data["calcium_dict"]
valve_dict = data["valve_dict"]

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

    # Find phase and ca values around events
    phases = []
    calcium_resp = []
    onset_resp = []
    for i in range(len(whiff_onsets)):
        start_idx = whiff_onsets[i]
        phase_value = np.mean(phase_peaks_dict[animal][start_idx:start_idx + 75])  # Mean phase in the 75 ms following pulse onset (75 samples at 1kHz)
        phases.append(phase_value)

        # index of the value  of resp_ts that is the closest to stim_ts[start_idx]
        index = np.absolute(ca_ts-new_valve_ts[start_idx]).argmin()
        resp = calcium_dict[animal][index:index + 40] # calcium response in the 4s following the pulse onset (10Hz sampling)
        calcium_resp.append(resp)
        onset_resp.append(index)

    data = {
        'whiff_onset': onset_resp,
        'phase': phases,
        'calcium': calcium_resp
        }
    all_data.append(data)

animals = list(valve_dict.keys())
final_dict = dict(zip(animals, all_data))

with open("Data/ca_resp_to_pulse_data.pkl","wb") as f:
    pickle.dump(final_dict,f)
