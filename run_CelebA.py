import utils

# parameters
command_template = 'python train_celeba_bc_ex.py --bb 1 --exp_name {} --task {} --eval_mode {}'

p1 = ['421_6_MI']

p2 = ['makeup', 'blonde']

# p3 = ['unbiased']
p3 = ['unbiased', 'conflict', 'unbiased_ex', 'conflict_ex']

trials = 10
for _ in range(trials):
    utils.run(command_template, "premium", 1, p1, p2, p3)
    # utils.run(command_template, "flexible", 1, p1, p2, p3)

