import os
from itertools import product
import time

def get_running_jobs():
    import os
    res = os.popen('squeue -u jiazli').read().splitlines()
    return res


def run(command_template, qos, gpu, *args, check=False):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    # if not os.path.exists('errors'):
    #     os.makedirs('errors')

    l = len(args)
    job_name_template = '{}'
    for _ in range(l-1):
        job_name_template += '-{}'
    for a in product(*args):
        command = command_template.format(*a)
        job_name = job_name_template.format(*a)
        bash_file = '{}.sh'.format(job_name)
        with open( bash_file, 'w' ) as OUT:
            OUT.write('#!/bin/bash\n')
            OUT.write('#SBATCH --job-name={} \n'.format(job_name))
            OUT.write('#SBATCH --ntasks=1 \n')
            OUT.write('#SBATCH --account=other \n')
            OUT.write(f'#SBATCH --qos={qos} \n')
            OUT.write('#SBATCH --partition=ALL \n')
            OUT.write('#SBATCH --cpus-per-task=4 \n')
            OUT.write(f'#SBATCH --gres=gpu:{gpu} \n')
            OUT.write('#SBATCH --mem={}G \n'.format(32 * gpu))
            OUT.write('#SBATCH --time=5-00:00:00 \n')
            # OUT.write('#SBATCH --exclude=vista[01,02,03,12,09,04,14,20] \n')
            OUT.write('#SBATCH --output=outputs/{}.out \n'.format(job_name))
            OUT.write('#SBATCH --error=outputs/{}.out \n'.format(job_name))
            OUT.write('source ~/.bashrc\n')
            OUT.write('echo $HOSTNAME\n')
            OUT.write('echo $HOSTNAME\n')
            OUT.write('echo $HOSTNAME\n')
            OUT.write('echo $HOSTNAME\n')
            OUT.write('echo $HOSTNAME\n')
            OUT.write('echo $HOSTNAME\n')
            OUT.write('echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES\n')
            OUT.write('conda activate pytorch\n')
            OUT.write(command)
        qsub_command = 'sbatch {}'.format(bash_file)
        os.system( qsub_command )
        os.system('rm -f {}'.format(bash_file))
        print( qsub_command )
        print( 'Submitted' )

        while check:
            if len(get_running_jobs()) <= 26:
                break
            else:
                time.sleep(60)