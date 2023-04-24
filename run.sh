python train_celeba_bc.py --bb 1 --task makeup --seed 1

python train_celeba_bc.py --bb 1 --task blonde --seed 1

python train_celeba_bc_ex.py --bb 1 --task blonde --eval_mode unbiased
python train_celeba_bc_ex.py --bb 1 --task blonde --eval_mode conflict 
python train_celeba_bc_ex.py --bb 1 --task blonde --eval_mode unbiased_ex
python train_celeba_bc_ex.py --bb 1 --task blonde --eval_mode conflict_ex

python train_celeba_bc_ex.py --bb 1 --task makeup --eval_mode unbiased
python train_celeba_bc_ex.py --bb 1 --task makeup --eval_mode conflict 
python train_celeba_bc_ex.py --bb 1 --task makeup --eval_mode unbiased_ex
python train_celeba_bc_ex.py --bb 1 --task makeup --eval_mode conflict_ex