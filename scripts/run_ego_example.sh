## Script to run all four environments
for VARIABLE in {1..10}
do
./main_egocentric.py with epochs=50000 name="main_ego" env="steplights" dev="False"
done

#for VARIABLE in {1..10}
#do
#./main_egocentric.py with epochs=50000 name="obs" env="obs" dev="False"
#done

#for VARIABLE in {1..10}
#do
#./main_egocentric.py with epochs=50000 name="obsint" env="obs_int" dev="False"
#done

#for VARIABLE in {1..10}
#do
#./main_egocentric.py with epochs=50000 name="int" env="int" dev="False"
#done