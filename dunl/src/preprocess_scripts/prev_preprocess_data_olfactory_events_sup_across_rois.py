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

    animals = ['HW1', 'HW4', 'Sphinx']

    for animal in animals:
        calcium_responses = data["calcium_dict"][animal].values.T
        calcium_responses = (calcium_responses - calcium_responses.min()) / (calcium_responses.max() - calcium_responses.min())
        
        valve = data["valve_dict"][animal]
        whiff_onsets = np.where(np.diff(valve) > 0)[0]
                
        valve_ts = np.arange(0, len(valve) / 1000, 0.001)  # 1 kHz sampling
        ca_ts = np.arange(0, calcium_responses.shape[1] / 10, 0.1)  # 10 Hz sampling
        
        # Find event indices in terms of calcium ref frame
        onset_resp = [
            np.abs(ca_ts - valve_ts[start_idx]).argmin()
            for start_idx in whiff_onsets
            if np.abs(ca_ts - valve_ts[start_idx]).argmin() < calcium_responses.shape[1] - params["kernel_length"]
        ]
        onset_resp=np.array(onset_resp)
            
        num_glom = calcium_responses.shape[0]
        trial_length = calcium_responses.shape[1]
        num_trials = 1

        trials_all = []
        y = np.zeros((num_trials, num_glom, trial_length))
        baseline = np.zeros((num_trials, num_glom, 1))
                
        for i in range(num_glom):
            y[0, i, :] =  calcium_responses[i]
        
        baseline[0, 0, 0] = 0
        
        trial_data = {
                "type": 0,
                "event0_onsets": onset_resp
                }
        trials_all.append(trial_data)

        data_dict = {
            'y': y,
            'a': baseline,
            'kernel_num' : params["kernel_num"]
        }

        for idx, trial_info in enumerate(trials_all):
            data_dict[f'trial{idx}'] = trial_info

        print(data_dict)
                            
        save_path = f'sparseness\Data\general_format_processed_{animal}.npy'
        torch.save(data_dict, save_path)
        print(f"Processed data saved to {save_path}.")
        
        print("Data structure:")
        print(" - y: Calcium response, shape (num_trials, num_glom, trial_length)")
        print(" - a: Baseline activity, shape (num_trials, num_glom, 1)")
        print(" - key kernel_num: number of kernels.")
        print(" - trial#: Metadata for each trial (event_onsets, type)")
    

if __name__ == '__main__':
    main()
