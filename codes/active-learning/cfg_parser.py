import numpy as np
import os
import subprocess
import shutil
import matplotlib.pyplot as plt
import datetime

mvsetpoints = np.concatenate([np.linspace(0.1, 2, 50), np.linspace(2.1,10,50)])

class Cfg:
    def __init__(self):
        pass

    def extract_from_text(self, lines):
        
        # lines[0] should be BEGIN_CFG
        # lines[1] should be END_CFG

        assert 'BEGIN_CFG' in lines[0]
        assert 'END_CFG' in lines[-1]

        self.lines = lines
        self.mv_grade = float(lines[-2].split()[-1])
        
        pass

def extract_cfgs_from_file(file='preselected.cfg'):

    cfg_list = []

    with open(file,'r') as f:
        lines = f.readlines()
    
    for cline, line in enumerate(lines):
        if 'BEGIN_CFG' in line:
            start_ind = cline
        
        if 'END_CFG' in line:
            end_ind = cline
            tmp = Cfg()
            tmp.extract_from_text(lines[start_ind:end_ind+1])
            cfg_list.append(tmp)
    
    return cfg_list

def select_cfg_by_mv(cfg_list, mvsetpoints = np.concatenate([np.linspace(0.1, 2, 20), np.linspace(2.1,10,20)])):
    '''From list of Cfg objects, get ones whose mv grades are closest to the setpoint mv grades 
    '''

    mv_list = np.array([cfg.mv_grade for cfg in cfg_list])

    arg_list = []

    for mv in mvsetpoints:
        arg = np.argmin((mv_list - mv)**2)
        arg_list.append(arg)
        print(f'MV setpoint: {mv}, cfg.mv: {cfg_list[arg].mv_grade}')

    result_cfg = [ cfg_list[i] for i in arg_list  ]

    return result_cfg

def write_mvsetpoint_cfg_to_folder(output_folder='selected_cfgs', mvsetpoints=np.concatenate([np.linspace(0.1, 2, 20), np.linspace(2.1,10,20)])):
    cfg_list = extract_cfgs_from_file()
    cfg_list = select_cfg_by_mv(cfg_list=cfg_list, mvsetpoints=mvsetpoints)

    os.makedirs(output_folder, exist_ok=True)

    with open(f'{output_folder}/preselected.cfg','w') as f:
        for cfg in cfg_list:
            f.writelines(cfg.lines)
            f.write('\n')

    pass

def generate_vasp_inputs(selected_cfgs_file = 'selected_cfgs/preselected.cfg', output_dir = 'selected_cfgs/vasp', incar_file='pb_vasp/INCAR', potcar_file='pb_vasp/POTCAR'):

    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([f"mlp convert --output_format=poscar {selected_cfgs_file} {output_dir}/"], shell=True)

    vasp_poscar_configs = []

    for tmp in os.listdir(output_dir):
        if os.path.isdir(f'{output_dir}/{tmp}'):
            continue
        else:
            vasp_poscar_configs.append(tmp)
    vasp_poscar_configs = sorted(vasp_poscar_configs)

    assert os.path.exists(incar_file)
    assert os.path.exists(potcar_file)

    for cfg in vasp_poscar_configs:
        os.makedirs(f'{output_dir}/vasp_{cfg}', exist_ok=True)
        shutil.move( f'{output_dir}/{cfg}' ,f'{output_dir}/vasp_{cfg}/POSCAR')
        shutil.copy(incar_file, f'{output_dir}/vasp_{cfg}/INCAR')
        shutil.copy(potcar_file, f'{output_dir}/vasp_{cfg}/POTCAR')
    
def run_vasp_files(folder = 'selected_cfgs/vasp', almtp_file = '/home/yimeng/pb-dpmodel/mlip3_ym_test/mlip_pythontools/mlip_ym_all_train/mtp_18/pot.almtp'):

    vasp_folders = get_sorted_folders(dir=folder)

    cwd = os.getcwd()
    for vasp in vasp_folders:
        print(f'Starting run: {vasp}', flush=True)
        os.chdir(f'{folder}/{vasp}')
        # Run vasp 
        subprocess.run(["mpirun -np 16 vasp_std"], shell=True)

        # Run MLIP to first convert outcar to mlip_cfg
        subprocess.run([f'mlp convert --input_format=outcar OUTCAR mlip_test.cfg'], shell=True)
        # run test 
        output_log = subprocess.run([f"mlp check_errors --log=test_eq_cfg.log {almtp_file} mlip_test.cfg "], capture_output=True, text=True, shell=True)

        with open(f'test.log','w') as f:
            f.writelines(output_log.stdout)

        # change back to common directory 
        os.chdir(cwd)

    print(f'Completed at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


    pass

def plot_err_vs_expolation_grade(folder = 'selected_cfgs/vasp'):
    ''' not implemented! 
    '''

    assert NotImplementedError

    vasp_folders = get_sorted_folders(folder)

    energy_diff = []
    forces_rms = []

    cfg_mv = select_cfg_by_mv(extract_cfgs_from_file(), mvsetpoints=mvsetpoints)
    cfg_mv = [i.mv_grade for i in cfg_mv]

    for vasp in vasp_folders:

        print(f'{vasp}')
        # read test_eq_cfg.log.0
        assert os.path.exists(f'{folder}/{vasp}/test_eq_cfg.log.0')
        with open(f'{folder}/{vasp}/test_eq_cfg.log.0','r') as f:
            lines = f.readlines()[0]
        
        lines = lines.split()
        energy_diff.append(float(lines[5].split(':')[-1]))
        forces_rms.append(float(lines[8]))

    
    plt.scatter(cfg_mv, energy_diff)
    plt.ylabel(r'abs($E - E_{ref}$)')
    plt.xlabel(f'Extrapolation grade')
    plt.tight_layout()
    plt.savefig('exgrade_ene.pdf')
    plt.close()

    plt.scatter(cfg_mv, forces_rms)
    plt.ylabel(r'$|| \vec{F} - \vec{F}_{ref} ||_{2}$')
    plt.xlabel(f'Extrapolation grade')
    plt.tight_layout()
    plt.savefig('exgrade_rmseforce.pdf')
    plt.close()

    pass

def plot_train_history(mlp_out_file):

    with open(f'{mlp_out_file}','r') as f:
        lines = f.readlines()

    f = []
    for line in lines:
        if 'BFGS iter' in line:
            tmp = line.split('=')[-1]
            f.append(float(tmp))

    f = np.array(f)
    fig = plt.figure()
    plt.plot(np.arange(len(f)), f)
    plt.xlabel(f'BFGS Iter')
    plt.ylabel(f'f [-]')
    plt.yscale(f'log')
    plt.title(f'Loss history')
    plt.tight_layout()
    
    return fig 

def get_sorted_folders(dir, startswith='vasp'):
    tmp = os.listdir(dir)
    valid_dirs = []
    sorted_args = []
    for t in tmp:
        if os.path.isdir(f'{dir}/{t}'):
            if t.startswith(startswith):
                sorted_args.append(int(t.split('_')[-1]))
                valid_dirs.append(t)

    sorted_args = np.argsort(sorted_args)

    result_sorted_dirs = []
    for arg in sorted_args:
        result_sorted_dirs.append(valid_dirs[arg])

    return result_sorted_dirs

if __name__ == "__main__":

    # write_mvsetpoint_cfg_to_folder(mvsetpoints=mvsetpoints)

    # generate_vasp_inputs()
    # run_vasp_files()

    plot_err_vs_expolation_grade(folder = 'selected_cfgs/vasp')
    pass
