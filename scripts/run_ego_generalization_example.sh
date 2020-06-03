## Script to run all generalization environments

#Generalization with the size of the environment
#for VARIABLE in {1..2}
#do
#./main_egocentric.py with epochs=100000 name="main_ego_gen_size" env="pushbuttons" dev="False" randomize_size="True" a_size=4
#done

#With many buttons...
for VARIABLE in {1..2}
do
./main_egocentric.py with epochs=100000 name="main_ego_gen_buttons" env="pushmanybuttons" dev="False" env_dim_max=5 a_size=4
done
