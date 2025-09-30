import torch
import numpy as np
import argparse
import pickle

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="sparseness/Data/ca_resp_to_pulse_data.pkl", 
    )
    parser.add_argument(
        "--kernel-num",
        type=int,
        help="number of convolutional kernels",
        default=1,
    )

    args = parser.parse_args()
    params = vars(args)

    return params



def main():
    params = init_params()

    print("data {} is being processed!".format(params["data_path"]))

    with open(params["data_path"],"rb") as f:
        data= pickle.load(f)

    y_all = []
    a_all = []
    trials_all =[]
    phases_all = []

    for animal in data.keys():

        whiff_onsets = data[animal]['whiff_onset']
        phases = data[animal]['phase']
        calcium_responses = data[animal]['calcium'] 

        num_trials = len(whiff_onsets)
        trial_length = 40
        num_neurons = 1  # 1 calcium response channel (bc resp is avged over ROIs)

        y = np.zeros((num_trials, num_neurons, trial_length))
        phase_data = np.zeros((num_trials, 1))
        baseline = np.zeros((num_trials, num_neurons, 1))

        for trial in range(num_trials):
            calcium_resp = np.array(calcium_responses[trial])

            # Handle unexpected dimensions or empty arrays
            if calcium_resp.ndim == 0 or calcium_resp.size == 0:
                print(f"Warning: Trial {trial} has empty or malformed calcium response.")
                calcium_resp = np.zeros(trial_length)
            elif len(calcium_resp) < trial_length:
                calcium_resp = np.pad(calcium_resp, (0, trial_length - len(calcium_resp)), mode='constant')
            
            y[trial, 0, :] = calcium_resp[:trial_length]

            phase = phases[trial]
            phase_data[trial, 0] = phases[trial]


            # Classify the event type based on phase
            """if phase.min() <= phase <= np.pi:
                event_type = 0  
            elif np.pi < phase <= phase.max():
                event_type = 1  
            else:
                raise ValueError(f"Phase value out of bounds: {phase}")"""
            event_type = 1

            
            """trial_info = {
                "type": event_type,
                "event0_onsets": [1] if event_type == 0 else [],
                "event1_onsets": [1] if event_type == 1 else [],
            }"""
            trial_info = {
                "type": event_type,
                "event0_onsets": [0]
            }

            trials_all.append(trial_info)
            phases_all.append(phase)

            # Calculate baseline as the mean response during the pulse (begins at whiff onset and stops 50ms after)
            # Assuming the first 100 ms is a baseline window for calcium data
            baseline[trial, 0, 0] = np.mean(calcium_resp[:1])
            #baseline[trial,0,0] =0.0067

            #event_onsets[trial].append(whiff_onsets[trial])

        y_all.append(y)
        a_all.append(baseline)

    """y = np.concatenate(y_all)
    a = np.concatenate(a_all)"""
    y = y_all[0]
    a = a_all[0]

    data_dict = {
        'y': y,
        'a': a,
        'phase': phases_all,
        'kernel_num' : params["kernel_num"]
    }

    for idx, trial_info in enumerate(trials_all):
        data_dict[f'trial{idx}'] = trial_info


    save_path = 'sparseness/Data/processed_ca_resp_to_pulse_data.npy'
    torch.save(data_dict, save_path)
    print(f"Processed data saved to {save_path}.")

    print("Data structure:")
    print(" - y: Calcium responses, shape (num_trials, num_neurons, trial_length)")
    print(" - a: Baseline activity, shape (num_trials, num_neurons, 1)")
    print(" - phase: Mean phase values per trial, shape (num_trials)")
    print(" - key kernel_num: number of kernels.")
    print(" - trial#: Metadata for each trial (event_onsets, type)")

if __name__ == '__main__':
    main()
