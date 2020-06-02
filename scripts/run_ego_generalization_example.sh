## Script to run all generalization environments

#for VARIABLE in {1..5}
#do
#./main_egocentric.py with epochs=200000 name="main_ego_diag" env="pushbuttons_cardinal" dev="False"
#done

#./main_egocentric.py with epochs=20000 name="main_ego_gen_size" env="pushbuttons" dev="True" randomize_size="True"
for VARIABLE in {1..5}
do
./main_egocentric.py with epochs=50000 name="main_ego_gen_size" env="pushbuttons" dev="False" randomize_size="True"
done

#for VARIABLE in {1..5}
#do
#./main_egocentric.py with epochs=50000 name="main_ego_gen_buttons" env="pushmanybuttons" dev="False" randomize_size="False" env_dim_max=5
#done
