## Script to run all four environments
#for VARIABLE in {1..10}
#do
#./main_confounding.py with epochs=50000 name="simple_example_confounded" env="confounded" dev="False"
#done

for VARIABLE in {1..10}
do
./main_confounding.py with epochs=50000 name="simple_example_obs" env="obs" dev="False"
done

#for VARIABLE in {1..10}
#do
#./main_confounding.py with epochs=50000 name="simple_example_obsint" env="obs_int" dev="False"
#done

#for VARIABLE in {1..10}
#do
#./main_confounding.py with epochs=50000 name="simple_example_int" env="int" dev="False"
#done