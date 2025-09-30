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
        default="Data/ca_resp_to_paired_pulses_data.pkl", 
    )
    parser.add_argument(
        "--kernel-num",
        type=int,
        help="number of convolutional kernels",
        default=4,
    )

    args = parser.parse_args()
    params = vars(args)

    return params

def get_trial_type(phase1, phase2):
    if 0 <= phase1 < np.pi and 0 <= phase2 < np.pi:
            return 0 
    elif 0 <= phase1 < np.pi and np.pi <= phase2 < 2 * np.pi:
            return 1 
    elif np.pi <= phase1 < 2 * np.pi and 0 <= phase2 < np.pi:
            return 2
    elif np.pi <= phase1 < 2 * np.pi and np.pi <= phase2 < 2 * np.pi:
            return 3
    else:
            raise ValueError(f'Phase value out of bounds: {phase1}, {phase2}')

def main():
    params = init_params()

    print("data {} is being processed!".format(params["data_path"]))

    with open(params["data_path"],"rb") as f:
        data= pickle.load(f)

    y_all = []
    a_all = []
    trials_all =[]

    for animal in data.keys():

        whiff1_onsets = data[animal]['whiff1_onset']
        whiff2_onsets = data[animal]['whiff2_onset']
        phases1 = data[animal]['phase1']
        phases2 = data[animal]['phase2']
        calcium_responses = data[animal]['calcium'] 

        num_trials = len(whiff1_onsets)
        trial_length = max(len(arr) for arr in calcium_responses)
        num_neurons = 1  # 1 calcium response channel (bc resp is avged over ROIs)

        y = np.zeros((num_trials, num_neurons, trial_length))
        a = np.zeros((num_trials, num_neurons, 1))

        for i in range(num_trials):
            trial_type = get_trial_type(phases1[i], phases2[i])

            y[i, 0, :len(calcium_responses[i])] = calcium_responses[i]
            # TRY TO VARY NUMBER OF FRAMES IN TO TAKE INTO ACCOUNT IN THE BASELINE CALCULATION
            a[i,0,0] = np.mean(calcium_responses[i][-5:])

            trial_info = {
                 'type':trial_type,
                 'event0_onsets':  [whiff2_onsets[i]] if (0 <= phases1[i] < np.pi and 0 <= phases2[i] < np.pi) else [],
                 'event1_onsets':  [whiff2_onsets[i]] if (0 <= phases1[i] < np.pi and np.pi <= phases2[i] < 2 * np.pi) else [],
                 'event2_onsets':  [whiff2_onsets[i]] if (np.pi <= phases1[i] < 2 * np.pi and 0 <= phases2[i] < np.pi) else [],
                 'event3_onsets':  [whiff2_onsets[i]] if (np.pi <= phases1[i] < 2 * np.pi and  np.pi <= phases2[i] < 2 * np.pi) else []
            }

            trials_all.append(trial_info)

        y_all.append(y)
        a_all.append(a)

    data_dict = {
        'y': np.concatenate(y_all,axis=0),
        'a': np.concatenate(a_all, axis=0),
        'kernel_num' : params["kernel_num"]
    }

    for idx, trial_info in enumerate(trials_all):
        data_dict[f'trial{idx}'] = trial_info
    
    save_path = 'Data\processed_ca_resp_to_paired_pulses_2_data.npy'
    torch.save(data_dict, save_path)
    print(f"Processed data saved to {save_path}.")

    print("Data structure:")
    print(" - y: Calcium responses, shape (num_trials, num_neurons, trial_length)")
    print(" - a: Baseline activity, shape (num_trials, num_neurons, 1)")
    print(" - key kernel_num: number of kernels.")
    print(" - trial#: Metadata for each trial (event_onsets, type)")

if __name__ == '__main__':
    main()
