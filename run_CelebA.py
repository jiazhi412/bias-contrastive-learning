import utils

# parameters
command_template = 'python {} --exp_name {} --task {} --eval_mode {}'

# p0 = ['train_celeba_bc_ex.py']
# p0 = ['train_celeba_lff_ex.py', 'train_celeba_end_ex.py', 'train_celeba_di_ex.py', 'train_celeba_lnl_ex.py']
# p0 = ['train_celeba_lff_ex.py']
# p0 = ['train_celeba_bc_ex.py', 'train_celeba_end_ex.py']
# p0 = ['train_celeba_di_ex.py', 'train_celeba_lnl_ex.py']
p0 = ['train_celeba_bc_ex.py', 'train_celeba_lff_ex.py', 'train_celeba_end_ex.py', 'train_celeba_di_ex.py', 'train_celeba_lnl_ex.py']

p1 = ['Paper_MI']

p2 = ['makeup', 'blonde']

# p3 = ['unbiased']
# p3 = ['unbiased', 'unbiased_ex']
p3 = ['unbiased', 'conflict', 'unbiased_ex', 'conflict_ex']

trials = 4
for _ in range(trials):
    utils.run(command_template, "premium", 1, p0, p1, p2, p3)
    # utils.run(command_template, "flexible", 1, p0, p1, p2, p3)