for i in {355394..355409}
do
scancel $i
done
squeue --sort=-i -o "%8i %40j %4t %10u %10q %10a %10g %10P %10Q %5D %11l %11L %R" | grep 'jiazli'