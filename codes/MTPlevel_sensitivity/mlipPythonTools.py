import os
import numpy as np 
import subprocess
import shutil 
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

class mlipModel:
    def __init__(self, savedir:str):
        self.savedir = savedir
        os.makedirs(savedir, exist_ok=True)
        pass

    @property
    def training_outcars(self):
        return self._training_outcars

    @training_outcars.setter
    def training_outcars(self, values:list[str]):
        # check that the files exist

        for value in values:
            assert os.path.exists(value), f"File '{value}' does not exist"

        self._training_outcars = values

    @property
    def MTP_level(self):
        return self._MTP_level

    @MTP_level.setter
    def MTP_level(self, M:int):
        assert M in list(np.arange(6, 29, 2))
        self._MTP_level = M
        pass

    def create_training_config(self):

        if os.path.exists(f'{self.savedir}/train.cfg'):
            os.remove(f'{self.savedir}/train.cfg')

        n_config = []

        for count, outcar_file in enumerate(self.training_outcars):
            assert os.path.exists(outcar_file), f"File '{outcar_file}' does not exist"
            output_log = subprocess.run([f'mlp convert --input_format=outcar {outcar_file} {self.savedir}/card_{count}.cfg'], shell=True,capture_output=True, text=True)
            tmp = int(output_log.stdout.split()[1])
            print(f'OUTCAR {outcar_file} contains {tmp} configs', flush=True)
            n_config.append(tmp)

        print(f'Total number of configurations: {n_config}')

        # concat all the files together and delete 
        subprocess.run([f'cat {self.savedir}/card_*.cfg >> {self.savedir}/train.cfg'], shell=True)

        for count in range(len(self.training_outcars)):
            os.remove(f'{self.savedir}/card_{count}.cfg')

        return n_config

    def train(self, mpi:int=8):

        print(f'Starting training: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. Num of MPI: {mpi}', flush=True)

        mtp_level_str = f'{self.MTP_level}'.zfill(2)

        # copy over MTP template file over 
        almtp_file = os.path.join(os.environ["HOME"], f'/home/yimeng/mlip-3-main/MTP_templates/{mtp_level_str}.almtp')

        
        shutil.copy(almtp_file, f"{self.savedir}/{mtp_level_str}.almtp")

        # create config file from all the files in self.training_outcars
        # self.create_training_config()

        # no active learning. Use mpi for faster speed i.e. mpirun -np <N> mlp help

        # os.chdir(self.savedir)
        output_log = subprocess.run([f"mpirun  -np {mpi} mlp train {self.savedir}/{mtp_level_str}.almtp {self.savedir}/train.cfg --save_to={self.savedir}/pot.almtp --iteration_limit=100"], capture_output=True, text=True, shell=True)

        with open(f'{self.savedir}/train.log','w') as f:
            f.writelines(output_log.stdout)

        # os.chdir('../')

        # plot BFGS iteration f value from out file

        print(f'Training completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

        self.plot_train_history()
        self.save_to_file()

        pass

    def create_test_data_cfg(self):
        '''Create configs for neq and eq data for 32/108 atom configurations and save to self.savedir/neq_test.cfg and self.savedir/eq_test.cfg
        '''
        neq32 = [] 
        for scale_factor in [0.9, 0.92, 0.94, 0.96, 0.98, 1.02, 1.04, 1.06, 1.08]:
            tmp = f'scale-{scale_factor:1.3f}'
            for i in range(5):
                neq32.append( f'../../OUTCARS/pb_testdata/32/neq/{tmp}/' + f'{i}'.zfill(6) + '/OUTCAR' )

        neq108 = []
        for scale_factor in [0.9, 0.92, 0.94, 0.96, 0.98, 1.02, 1.04, 1.06, 1.08]:
            tmp = f'scale-{scale_factor:1.3f}'
            for i in range(5):
                neq32.append( f'../../OUTCARS/pb_testdata/108/neq/{tmp}/' + f'{i}'.zfill(6) + '/OUTCAR' )

        # convert to single test file 
        neq = neq32 + neq108

        for count, outcar_file in enumerate(neq):
            print(f'neq count: {count} out of {len(neq)}', flush=True)
            assert os.path.exists(outcar_file), f"File '{outcar_file}' does not exist"
            subprocess.run([f'mlp convert --input_format=outcar {outcar_file} {self.savedir}/card_{count}.cfg'], shell=True)

        # concat all the files together and delete 
        subprocess.run([f'cat {self.savedir}/card_*.cfg >> {self.savedir}/neq_test.cfg'], shell=True)

        for count in range(len(neq)):
            os.remove(f'{self.savedir}/card_{count}.cfg')

        # eq dataset
        eq32  = ['../../OUTCARS/pb_testdata/32/eq/scale-1.000/' + f'{i}'.zfill(6) + '/OUTCAR' for i in range(21) ]
        eq108 = ['../../OUTCARS/pb_testdata/108/eq/scale-1.000/' + f'{i}'.zfill(6) + '/OUTCAR' for i in range(21) ]
        eq = eq32 + eq108
        for count, outcar_file in enumerate(eq):
            print(f'eq count: {eq} out of {len(eq)}', flush=True)
            assert os.path.exists(outcar_file), f"File '{outcar_file}' does not exist"
            subprocess.run([f'mlp convert --input_format=outcar {outcar_file} {self.savedir}/card_{count}.cfg'], shell=True)

        # concat all the files together and delete 
        subprocess.run([f'cat {self.savedir}/card_*.cfg >> {self.savedir}/eq_test.cfg'], shell=True)

        for count in range(len(eq)):
            os.remove(f'{self.savedir}/card_{count}.cfg')

        pass

    def test(self):
        ''' Calculate test metrics (RMSE)
        '''

        if (os.path.exists(f'{self.savedir}/eq_test.cfg') == False) or (os.path.exists(f'{self.savedir}/neq_test.cfg') == False):
            print(f'Creating test data configurations: eq_test.cfg and neq_test.cfg')
            self.create_test_data_cfg()

        
        print(f'Starting test {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        # eq test
        output_log = subprocess.run([f"mpirun -np 4 mlp check_errors --log={self.savedir}/test_eq_cfg.log {self.savedir}/pot.almtp {self.savedir}/eq_test.cfg "], capture_output=True, text=True, shell=True)

        with open(f'{self.savedir}/test_eq.log','w') as f:
            f.writelines(output_log.stdout)

        print(f'Finished eq test {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        # neq test
        output_log = subprocess.run([f"mpirun  -np 4 mlp check_errors --log={self.savedir}/test_neq_cfg.log {self.savedir}/pot.almtp {self.savedir}/neq_test.cfg "], capture_output=True, text=True, shell=True)

        with open(f'{self.savedir}/test_neq.log','w') as f:
            f.writelines(output_log.stdout)

        print(f'Finished neq test {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        pass


    def get_train_history(self) -> np.ndarray:
        '''Get BFGS iter loss f
        '''

        assert os.path.exists(f'{self.savedir}/train.log'), f"Training log of MLIP {self.savedir}/train.log' does not exist. Do self.train() first"

        with open(f'{self.savedir}/train.log','r') as f:
            lines = f.readlines()

        f = []
        for line in lines:
            if 'BFGS iter' in line:
                tmp = line.split('=')[-1]
                f.append(float(tmp))

        f = np.array(f)
        return f

    def plot_train_history(self):

        f = self.get_train_history()
        plt.figure()
        plt.plot(np.arange(len(f)), f)
        plt.xlabel(f'BFGS Iter')
        plt.ylabel(f'f [-]')
        plt.yscale(f'log')
        plt.title(f'Loss history')
        plt.tight_layout()
        plt.savefig(f'{self.savedir}/train_history.pdf')
        plt.close()

        pass

    def save_to_file(self):
        '''Save copy of this object to file 
        '''
        with open(f'{self.savedir}/mlip_model.pkl','wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(savedir:str) -> 'mlipModel':
        '''Returns copy of mlipModel
        '''
        
        assert os.path.exists(f'{savedir}/mlip_model.pkl'), f"File '{savedir}/mlip_model.pkl' does not exist"

        with open(f'{savedir}/mlip_model.pkl','rb') as f:
            model = pickle.load(f)

        return model


if __name__ == "__main__":

    # mlip_model = mlipModel(savedir='test_mlip')
    # mlip_model.training_outcars = ['']

    pass