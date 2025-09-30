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
        default="sparseness/Data/animals_data_processed.pkl", 
    )
    parser.add_argument(
        "--kernel-num",
        type=int,
        help="number of convolutional kernels",
        default=1,
    )
    parser.add_argument(
        "--kernel-length",
        type=int,
        help="kernel length",
        default=20,
    )

    args = parser.parse_args()
    params = vars(args)

    return params

def main():
    params = init_params()

    print("data {} is being processed!".format(params["data_path"]))

    with open(params["data_path"],"rb") as f:
        data= pickle.load(f)

    #animals = ['HW1', 'HW4', 'Sphinx']
    animals = ['HW1']

    onsets_all = []
    calcium_all = []

    offset = 0
    for animal in animals:
        #calcium_responses = data["calcium_dict"][animal].mean(axis=1).to_numpy()
        calcium_responses = data["calcium_dict"][animal][0]
        calcium_responses = (calcium_responses - calcium_responses.min()) / (calcium_responses.max() - calcium_responses.min())
        
        valve = data["valve_dict"][animal]
        whiff_onsets = np.where(np.diff(valve) > 0)[0]
        
        valve_ts = np.arange(0, len(valve) / 1000, 0.001)  # 1 kHz sampling
        ca_ts = np.arange(0, len(calcium_responses) / 10, 0.1)  # 10 Hz sampling
        
        # Find event indices in terms of calcium ref frame and add offset
        onset_resp = [
            np.abs(ca_ts - valve_ts[start_idx]).argmin() + offset
            for start_idx in whiff_onsets
            if np.abs(ca_ts - valve_ts[start_idx]).argmin() < len(calcium_responses) - params["kernel_length"]
        ]
        
        onsets_all.append(onset_resp)
        calcium_all.append(calcium_responses)
        
        # Update offset for the next animal
        offset += len(calcium_responses)

    # Concatenate results
    onsets = np.concatenate(onsets_all)
    calcium = np.concatenate(calcium_all)
    
    num_glom = 1
    trial_length = len(calcium)
    num_trials = 1

    y = np.zeros((num_trials, num_glom, trial_length))
    baseline = np.zeros((num_trials, num_glom, 1))

    y[0, 0, :] =  calcium
    baseline[0, 0, 0] = 0
    trial_data = {
        "type": 0,
        "event0_onsets": onsets
        }

    data_dict = {
        'y': y,
        'a': baseline,
        'kernel_num' : params["kernel_num"]
    }

    data_dict[f'trial0'] = trial_data
    
    print(data_dict)

    save_path = 'sparseness\Data\general_format_processed.npy'
    torch.save(data_dict, save_path)
    print(f"Processed data saved to {save_path}.")
    
    print("Data structure:")
    print(" - y: Calcium response, shape (num_trials, num_glom, trial_length)")
    print(" - a: Baseline activity, shape (num_trials, num_glom, 1)")
    print(" - key kernel_num: number of kernels.")
    print(" - trial#: Metadata for each trial (event_onsets, type)")
    

if __name__ == '__main__':
    main()
