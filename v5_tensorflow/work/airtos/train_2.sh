# !bin/sh

echo 'Will start to run through each execution'

echo 'Running tests for Outlier #4'
../bin/python run_c51.py NUMIT=5000 LRATE=0.003 LAYERS=v2 ENV=macd ID=outlier_4_1 >> ./logs/outlier_4_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=5000 LRATE=0.0034 LAYERS=v2 ENV=macd ID=outlier_4_2 >> ./logs/outlier_4_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #5'
../bin/python run_c51.py NUMIT=5000 LRATE=0.0086 LAYERS=v3 ENV=macd ID=outlier_5_1 >> ./logs/outlier_5_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=5000 LRATE=0.0093 LAYERS=v3 ENV=macd ID=outlier_5_2 >> ./logs/outlier_5_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #6'
../bin/python run_c51.py NUMIT=150000 LRATE=0.006 LAYERS=v4 ENV=macd ID=outlier_6_1 >> ./logs/outlier_6_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=150000 LRATE=0.0064 LAYERS=v4 ENV=macd ID=outlier_6_2 >> ./logs/outlier_6_2.log
echo 'Finished execution'
