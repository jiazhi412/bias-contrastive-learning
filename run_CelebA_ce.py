import utils

# parameters
command_template = 'python {} --exp_name {} --task {} --eval_mode {} --p_bc {} --balance'

# p0 = ['train_celeba_bc_ex.py']
# p0 = ['train_celeba_lff_ex.py', 'train_celeba_end_ex.py', 'train_celeba_di_ex.py', 'train_celeba_lnl_ex.py']
# p0 = ['train_celeba_lff_ex.py']
# p0 = ['train_celeba_end_ex.py', 'train_celeba_lnl_ex.py']
# p0 = ['train_celeba_bc_ex.py', 'train_celeba_end_ex.py']
# p0 = ['train_celeba_di_ex.py']
# p0 = ['train_celeba_di_ex.py', 'train_celeba_lnl_ex.py']
p0 = ['train_celeba_bc_ex.py', 'train_celeba_lff_ex.py', 'train_celeba_end_ex.py', 'train_celeba_di_ex.py', 'train_celeba_lnl_ex.py']

p1 = ['Paper_CE_2']

p2 = ['makeup', 'blonde']

# p3 = ['unbiased']
p3 = ['unbiased', 'conflict']

p_bc = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

trials = 1
for _ in range(trials):
    utils.run(command_template, "premium", 1, p0, p1, p2, p3, p_bc)
    # utils.run(command_template, "flexible", 1, p0, p1, p2, p3, p_bc)

command_template = 'python {} --exp_name {} --task {} --eval_mode {} --p_bc {}'

p_bc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1]
# p_bc = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for _ in range(trials):
    utils.run(command_template, "premium", 1, p0, p1, p2, p3, p_bc)
    # utils.run(command_template, "flexible", 1, p0, p1, p2, p3, p_bc)