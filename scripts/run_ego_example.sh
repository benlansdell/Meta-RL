## Script to run all four environments

#for VARIABLE in {1..5}
#do
#./main_egocentric.py with epochs=200000 name="main_ego" env="pushbuttons" dev="False"
#done

for VARIABLE in {1..1}
do
./main_egocentric.py with epochs=50000 name="main_ego_diag" env="pushbuttons_cardinal" dev="True"
done