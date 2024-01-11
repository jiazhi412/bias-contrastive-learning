python train_celeba_bc.py --bb 1 --task makeup --seed 1

python train_celeba_end.py --task blonde 

python train_celeba_bc.py --bb 1 --task blonde --seed 1

python train_celeba_bc_ex.py --bb 1 --task blonde --eval_mode unbiased
python train_celeba_bc_ex.py --bb 1 --task blonde --eval_mode conflict 
python train_celeba_bc_ex.py --bb 1 --task blonde --eval_mode unbiased_ex
python train_celeba_bc_ex.py --bb 1 --task blonde --eval_mode conflict_ex

python train_celeba_bc_ex.py --bb 1 --task makeup --eval_mode unbiased
python train_celeba_bc_ex.py --bb 1 --task makeup --eval_mode conflict 
python train_celeba_bc_ex.py --bb 1 --task makeup --eval_mode unbiased_ex
python train_celeba_bc_ex.py --bb 1 --task makeup --eval_mode conflict_ex

python train_celeba_bc_ex.py --bb 1 --task blonde --eval_mode unbiased --p_bc 0.1 --balance


python train_biased_mnist_lff.py
python train_celeba_lff_ex.py --task makeup --eval_mode conflict_ex