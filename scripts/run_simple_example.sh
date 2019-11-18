## Script to run all four environments
#for VARIABLE in 1 .. 10
#do
#./main_confounding.py with epochs=50000 name="simple_example_confounded" env="confounded"
#done

#for VARIABLE in 1 .. 10
#do
#./main_confounding.py with epochs=50000 name="simple_example_obs" env="obs"
#done

#for VARIABLE in 1 .. 10
#do
#./main_confounding.py with epochs=50000 name="simple_example_obsint" env="obs_int"
#done

for VARIABLE in 1 .. 10
do
./main_confounding.py with epochs=50000 name="simple_example_int" env="int"
done