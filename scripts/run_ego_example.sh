## Script to run all four environments

#for VARIABLE in {1..5}
#do
#./main_egocentric.py with epochs=200000 name="main_ego" env="pushbuttons" dev="False"
#done

for VARIABLE in {1..2}
do
./main_egocentric.py with epochs=100000 name="main_ego_diag" env="pushbuttons_cardinal" dev="False" a_size=8 env_dim_max=5
done

#Test code:
./main_egocentric.py with epochs=100000 name="main_ego_diag" env="pushbuttons_cardinal" dev="True" a_size=8 env_dim_max=5