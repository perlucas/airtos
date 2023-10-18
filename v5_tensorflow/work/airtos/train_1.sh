# !bin/sh

echo 'Will start to run through each execution'

echo 'Running tests for Outlier #1'
../bin/python run_c51.py NUMIT=5000 LRATE=0.0058 LAYERS=v2 ENV=adx ID=outlier_1_1 >> ./logs/outlier_1_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=3000 LRATE=0.0058 LAYERS=v2 ENV=adx ID=outlier_1_2 >> ./logs/outlier_1_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #2'
../bin/python run_c51.py NUMIT=5000 LRATE=0.0086 LAYERS=v4 ENV=adx ID=outlier_2_1 >> ./logs/outlier_2_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=5000 LRATE=0.0093 LAYERS=v4 ENV=adx ID=outlier_2_2 >> ./logs/outlier_2_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #3'
../bin/python run_c51.py NUMIT=5000 LRATE=0.0086 LAYERS=v1 ENV=macd ID=outlier_3_1 >> ./logs/outlier_3_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=5000 LRATE=0.0094 LAYERS=v1 ENV=macd ID=outlier_3_2 >> ./logs/outlier_3_2.log
echo 'Finished execution'
