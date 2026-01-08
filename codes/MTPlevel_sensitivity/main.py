import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from mlipPythonTools import mlipModel

# on southpole use: python3 -m pip install matplotlib

def get_all_training_outcars():
    # get all training outcars, combining both eq and neq states

    eq108 = ['108eqOUTCARs/' + f'{i}'.zfill(6) for i in range(51) ]

    neq108 = []

    for scale_factor in [0.9, 0.92, 0.94, 0.96, 0.98, 1.02, 1.04, 1.06, 1.08]:
        tmp = f'108neqOUTCARs/scale-{scale_factor:1.3f}'
        for i in range(11):
            neq108.append( f'{tmp}/' + f'{i}'.zfill(6) )

    eq32 = ['32eqOUTCARs/' + f'{i}'.zfill(6) for i in range(51) ]    


    neq32 = []

    for scale_factor in [0.9, 0.92, 0.94, 0.96, 0.98, 1.02, 1.04, 1.06, 1.08]:
        tmp = f'32neqOUTCARs/scale-{scale_factor:1.3f}'
        for i in range(11):
            neq32.append( f'{tmp}/' + f'{i}'.zfill(6) )

    outcars = eq108 + neq108 + eq32 + neq32

    outcars = [f'../../OUTCARS/{i}/OUTCAR' for i in outcars]

    breakdown = {
        'eq108':len(eq108),
        'neq108':len(neq108),
        'eq32':len(eq32),
        'neq32':len(neq32),
    }

    return outcars, breakdown
    

def train_mlip_model():
    # initial test for MLIP model and varying the levels 


    #MTP_levels = [6, 12 ,18, 24, 28]
    MTP_levels = [8, 10, 14, 16, 20, 22]

    # train phase 
    for mtp_count, mtp_value in enumerate(MTP_levels):
        mlip_model = mlipModel(savedir=f'mlip_ym_all_train/mtp_{mtp_value}')
        outcar_files, _ = get_all_training_outcars()  
        mlip_model.training_outcars = outcar_files
        mlip_model.MTP_level = mtp_value
        shutil.copy(f'mlip_ym_all_train/mtp_6/train.cfg', mlip_model.savedir)
        mlip_model.train(mpi=16)

    pass


def setup_train_cfg():

    mlip_model = mlipModel(savedir=f'mlip_ym_all_train/mtp_6')
    outcar_files, breakdown = get_all_training_outcars()  
    mlip_model.training_outcars = outcar_files
    # mlip_model.MTP_level = mtp_value
    # create {model.savedir}/train.cfg file 
    mlip_model.training_outcars = outcar_files
    n_configs = mlip_model.create_training_config()

    nc = 0
    for key in breakdown.keys():

        v = np.sum(n_configs[nc:nc+breakdown[key]])
        print(f'dataset {key} has {v} configs')
        nc = nc + breakdown[key]

        pass

    mlip_model.save_to_file()

def test_mlip_model():

    MTP_levels = [6,8,10,12,14,16,18,20,22,24]
    for mtp_value in MTP_levels:
        mlip_model = mlipModel.load_from_file(savedir=f'mlip_ym_all_train/mtp_{mtp_value}')
        mlip_model.test()

def process_test_mlip_model():

    # read the test_eq.log and test_neq.log files and extract the RMSE for energy and for forces 
    MTP_levels = [6,8,10,12,14,16,18,20,22,24]
    # MTP_levels = [6]

    rmse_ener_neq = []
    rmse_forces_neq = []
    rmse_ener_eq = []
    rmse_forces_eq = []
    

    for mtp_value in MTP_levels:
        assert os.path.exists(f'mlip_ym_all_train/mtp_{mtp_value}/test_neq.log') 
        assert os.path.exists(f'mlip_ym_all_train/mtp_{mtp_value}/test_eq.log') 

        with open(f'mlip_ym_all_train/mtp_{mtp_value}/test_neq.log','r') as f:
            lines = f.readlines()
            rmse_ener_neq.append(float(lines[5].split()[-1]))
            rmse_forces_neq.append(float(lines[17].split()[-1]))

        with open(f'mlip_ym_all_train/mtp_{mtp_value}/test_eq.log','r') as f:
            lines = f.readlines()
            rmse_ener_eq.append(float(lines[5].split()[-1]))
            rmse_forces_eq.append(float(lines[17].split()[-1]))



    plt.plot(MTP_levels, rmse_ener_neq,'-o', label='neq test')
    plt.plot(MTP_levels, rmse_ener_eq,'-s', label='eq test')
    plt.legend()
    plt.title('RMSE config energy')
    plt.ylabel(f'RMSE')
    plt.xlabel('MTP level')
    plt.xticks(MTP_levels)
    plt.tight_layout()
    plt.savefig('energy_err_convergence.pdf')
    plt.close()

    plt.plot(MTP_levels, rmse_forces_neq,'-o', label='neq test')
    plt.plot(MTP_levels, rmse_forces_eq,'-s', label='eq test')
    plt.legend()
    plt.title('RMSE config forces')
    plt.ylabel(f'RMSE')
    plt.xlabel('MTP level')
    plt.xticks(MTP_levels)
    plt.tight_layout()
    plt.savefig('forces_err_convergence.pdf')
    plt.close()

    pass


def tmp_test():

    # model = mlipModel.load_from_file('mlip_ym_all_train/mtp_6')

    model = mlipModel('mlip_ym_all_train/mtp_6')

    # loss = model.get_train_history()
    model.plot_train_history()

    pass

def get_num_of_configs():

    pass



if __name__ == "__main__":

    # get_all_training_outcars()

    # setup_train_cfg()

    # train_mlip_model()

    # test_mlip_model()

    process_test_mlip_model()

    # tmp_test()

    pass