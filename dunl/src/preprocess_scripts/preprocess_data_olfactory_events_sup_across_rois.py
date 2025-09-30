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
    parser.add_argument(
        "--animal",
        type=str,
        help="animal name",
        default="Sphinx",
    )


    args = parser.parse_args()
    params = vars(args)

    return params

def main():
    params = init_params()

    print("data {} is being processed!".format(params["data_path"]))

    with open(params["data_path"],"rb") as f:
        data= pickle.load(f)

    animal = params["animal"]
    
    for i in range(data["calcium_dict"][animal].shape[1]):
        calcium_responses = data["calcium_dict"][animal][i]
        calcium_responses = (calcium_responses - calcium_responses.min()) / (calcium_responses.max() - calcium_responses.min())
        
        valve = data["valve_dict"][animal]
        whiff_onsets = np.where(np.diff(valve) > 0)[0]
        
        valve_ts = np.arange(0, len(valve) / 1000, 0.001)  # 1 kHz sampling
        ca_ts = np.arange(0, len(calcium_responses) / 10, 0.1)  # 10 Hz sampling
        
        # Find event indices in terms of calcium ref frame and add offset
        onset_resp = [
            np.abs(ca_ts - valve_ts[start_idx]).argmin()
            for start_idx in whiff_onsets
            if np.abs(ca_ts - valve_ts[start_idx]).argmin() < len(calcium_responses) - params["kernel_length"]
        ]
        onset_resp = np.array(onset_resp)
        
        num_glom = 1
        trial_length = len(calcium_responses)
        num_trials = 1

        y = np.zeros((num_trials, num_glom, trial_length))
        baseline = np.zeros((num_trials, num_glom, 1))

        y[0, 0, :] =  calcium_responses
        baseline[0, 0, 0] = 0
        trial_data = {
            "type": 0,
            "event0_onsets": onset_resp
            }

        data_dict = {
            'y': y,
            'a': baseline,
            'kernel_num' : params["kernel_num"]
        }

        data_dict[f'trial0'] = trial_data
        
        save_path = f'sparseness/Data/general_format_processed_roi{i}_{animal}.npy'
        torch.save(data_dict, save_path)
        print(f"Processed data saved to {save_path}.")
        
        print("Data structure:")
        print(" - y: Calcium response, shape (num_trials, num_glom, trial_length)")
        print(" - a: Baseline activity, shape (num_trials, num_glom, 1)")
        print(" - key kernel_num: number of kernels.")
        print(" - trial#: Metadata for each trial (event_onsets, type)")
    

if __name__ == '__main__':
    main()
