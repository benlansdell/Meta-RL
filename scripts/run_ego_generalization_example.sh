## Script to run all generalization environments

#for VARIABLE in {1..5}
#do
#./main_egocentric.py with epochs=200000 name="main_ego_diag" env="pushbuttons_cardinal" dev="False"
#done


./main_egocentric.py with epochs=20000 name="main_ego" env="pushbuttons" dev="True" randomize_size="True"
