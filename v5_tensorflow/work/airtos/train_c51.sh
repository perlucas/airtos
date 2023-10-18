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

echo 'Running tests for Outlier #7'
../bin/python run_c51.py NUMIT=5000 LRATE=0.0044 LAYERS=v3 ENV=rsi ID=outlier_7_1 >> ./logs/outlier_7_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=5000 LRATE=0.0054 LAYERS=v3 ENV=rsi ID=outlier_7_2 >> ./logs/outlier_7_2.log
echo 'Finished execution'

echo 'Running Additional Tests'
../bin/python run_c51.py NUMIT=4000 LRATE=0.006 LAYERS=v2 ENV=adx ID=extra_1 >> ./logs/extra_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=4000 LRATE=0.0096 LAYERS=v3 ENV=macd ID=extra_2 >> ./logs/extra_2.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=3000 LRATE=0.007 LAYERS=v4 ENV=macd ID=extra_3 >> ./logs/extra_3.log
echo 'Finished execution'