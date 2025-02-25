import pickle
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch

class Simulation_Dataset(Dataset):
    def __init__(self, data_length, scenario="1-1", data_folder=None, mode="train"):
        self.data_length = data_length
        self.scenarios = ["1-1","2-1","2-2","2-3","3-1","3-2", "4-1"]
        if data_folder is None:
            self.data_folder = "./data/simulation_data/"
        else:
            self.data_folder = data_folder

        person_info, sim_info = self._load_data(scenario)

        sim_info = self._filter_data(person_info, sim_info)


    def _filter_data(self, person_info, sim_info, time_limit=200):
        """
        Example of the input data:
            person_info (pd.DataFrame):
            PersonID	PersonType	Weight	Radius	MaxSpeed	GenTime	GenID	numGoal	GoalIDarray
            1	3	60.0000	0.210000	0.00000	0.10	3	1	3
            2	3	60.0000	0.210000	0.00000	0.10	4	1	4

            sim_info (pd.DataFrame):
            time	personID	posX	posY
            0.1	    1	        34.0	14.0
            0.1	    2	        61.0	11.0
            0.2	    1	        34.0	14.0
        """
        # remove the person who is standing (PersonType=1) in sim_info
        # Get the list of PersonIDs where PersonType == 1
        person_ids_to_remove = person_info[person_info['PersonType'] == 1]['PersonID']

        # Filter out rows in sim_info where personID matches those in person_ids_to_remove
        filtered_sim_info = sim_info[~sim_info['personID'].isin(person_ids_to_remove)]

        # Keep the sim_info rows where time is less than or equal to time_limit
        filtered_sim_info = filtered_sim_info[filtered_sim_info['time'] <= time_limit]
        return filtered_sim_info
            
    def _load_data(self, scenario):
        assert scenario in self.scenario
        if scenario == "4-1":
            path_sim = os.path.join(self.data_folder, "Scenario4-1/outputdata-1.6.2021/simulationLog_clean.csv")
            path_info = os.path.join(self.data_folder, "data/simulation_data/Scenario4-1/outputdata-1.6.2021/outputPersonInfo_clean.csv")
        else:
            path_sim = os.path.join(self.data_folder, f"Scenario{scenario}/output data-2.15(yamada@vri)/simulationLog.csv")
            path_info = os.path.join(self.data_folder,f"Scenario{scenario}/output data-2.15(yamada@vri)/outputPersonInfo.csv")
        person_info = pd.read_csv(path_info)
        sim_info = pd.read_csv(path_sim)
        return person_info, sim_info
        
    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]
        target_mask = self.mask_data[index:index+self.seq_length].copy()
        target_mask[-self.pred_length:] = 0. #pred mask for test pattern strategy
        s = {
            'observed_data': self.main_data[index:index+self.seq_length],
            'observed_mask': self.mask_data[index:index+self.seq_length],
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_length) * 1.0, 
            'feature_id': np.arange(self.main_data.shape[1]) * 1.0, 
        }

        return s
    def __len__(self):
        return len(self.use_index)

def get_dataloader(datatype,device,batch_size=8):
    dataset = Forecasting_Dataset(datatype,mode='train')
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Forecasting_Dataset(datatype,mode='valid')
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Forecasting_Dataset(datatype,mode='test')
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=0)

    scaler = torch.from_numpy(dataset.std_data).to(device).float()
    mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler