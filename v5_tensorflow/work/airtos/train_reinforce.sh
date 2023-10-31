# !bin/sh

echo 'Will start to run through each execution'

echo 'Running tests for Outlier #1'
../bin/python run_reinforce.py NUMIT=2000 LRATE=0.003 LAYERS=v1 ENV=macd ID=outlier_1_1 >> ./logs/outlier_1_1.log
echo 'Finished execution' 
../bin/python run_reinforce.py NUMIT=2000 LRATE=0.0024 LAYERS=v1 ENV=macd ID=outlier_1_2 >> ./logs/outlier_1_2.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=400 LRATE=0.003 LAYERS=v1 ENV=macd ID=outlier_1_3 >> ./logs/outlier_1_3.log
echo 'Finished execution'

echo 'Running tests for Outlier #2'
../bin/python run_reinforce.py NUMIT=100 LRATE=0.0058 LAYERS=v3 ENV=macd ID=outlier_2_1 >> ./logs/outlier_2_1.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=2000 LRATE=0.006 LAYERS=v3 ENV=macd ID=outlier_2_2 >> ./logs/outlier_2_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #3'
../bin/python run_reinforce.py NUMIT=50 LRATE=0.0058 LAYERS=v4 ENV=macd ID=outlier_3_1 >> ./logs/outlier_3_1.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1000 LRATE=0.005 LAYERS=v4 ENV=macd ID=outlier_3_2 >> ./logs/outlier_3_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #4'
../bin/python run_reinforce.py NUMIT=150 LRATE=0.003 LAYERS=v1 ENV=mas ID=outlier_4_1 >> ./logs/outlier_4_1.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1000 LRATE=0.0023 LAYERS=v1 ENV=mas ID=outlier_4_2 >> ./logs/outlier_4_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #5'
../bin/python run_reinforce.py NUMIT=1300 LRATE=0.0095 LAYERS=v3 ENV=mas ID=outlier_5_1 >> ./logs/outlier_5_1.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1300 LRATE=0.013 LAYERS=v3 ENV=mas ID=outlier_5_2 >> ./logs/outlier_5_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #6'
../bin/python run_reinforce.py NUMIT=600 LRATE=0.0086 LAYERS=v4 ENV=mas ID=outlier_6_1 >> ./logs/outlier_6_1.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1500 LRATE=0.0095 LAYERS=v4 ENV=mas ID=outlier_6_2 >> ./logs/outlier_6_2.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1500 LRATE=0.011 LAYERS=v4 ENV=mas ID=outlier_6_3 >> ./logs/outlier_6_3.log
echo 'Finished execution'

echo 'Running Additional Tests'
../bin/python run_reinforce.py NUMIT=1000 LRATE=0.01 LAYERS=v1 ENV=macd ID=extra_1 >> ./logs/extra_1.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1000 LRATE=0.01 LAYERS=v2 ENV=macd ID=extra_2 >> ./logs/extra_2.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1000 LRATE=0.01 LAYERS=v3 ENV=macd ID=extra_3 >> ./logs/extra_3.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1000 LRATE=0.01 LAYERS=v4 ENV=macd ID=extra_4 >> ./logs/extra_4.log
echo 'Finished execution'