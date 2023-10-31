# !bin/sh

echo 'Will start to run through each execution'

echo 'Running tests for Outlier #1'
../bin/python run_ppo.py NUMIT=35000000 LRATE=0.0044 LAYERS=v1 ENV=adx ID=outlier_1_1 >> ./logs/outlier_1_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=35000000 LRATE=0.006 LAYERS=v1 ENV=adx ID=outlier_1_2 >> ./logs/outlier_1_2.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=8000000 LRATE=0.0075 LAYERS=v1 ENV=adx ID=outlier_1_3 >> ./logs/outlier_1_3.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=7000000 LRATE=0.0073 LAYERS=v1 ENV=adx ID=outlier_1_4 >> ./logs/outlier_1_4.log
echo 'Finished execution'

echo 'Running tests for Outlier #2'
../bin/python run_ppo.py NUMIT=55000000 LRATE=0.0058 LAYERS=v2 ENV=adx ID=outlier_2_1 >> ./logs/outlier_2_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=7500000 LRATE=0.006 LAYERS=v2 ENV=adx ID=outlier_2_2 >> ./logs/outlier_2_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #3'
../bin/python run_ppo.py NUMIT=15000000 LRATE=0.003 LAYERS=v3 ENV=adx ID=outlier_3_1 >> ./logs/outlier_3_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=20000000 LRATE=0.0044 LAYERS=v3 ENV=adx ID=outlier_3_2 >> ./logs/outlier_3_2.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=35000000 LRATE=0.0044 LAYERS=v3 ENV=adx ID=outlier_3_3 >> ./logs/outlier_3_3.log
echo 'Finished execution'

echo 'Running tests for Outlier #4'
../bin/python run_ppo.py NUMIT=40000000 LRATE=0.0032 LAYERS=v4 ENV=adx ID=outlier_4_1 >> ./logs/outlier_4_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=40000000 LRATE=0.0038 LAYERS=v4 ENV=adx ID=outlier_4_2 >> ./logs/outlier_4_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #5'
../bin/python run_ppo.py NUMIT=2500000 LRATE=0.0072 LAYERS=v1 ENV=macd ID=outlier_5_1 >> ./logs/outlier_5_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=2000000 LRATE=0.0072 LAYERS=v1 ENV=macd ID=outlier_5_2 >> ./logs/outlier_5_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #6'
../bin/python run_ppo.py NUMIT=20000000 LRATE=0.003 LAYERS=v2 ENV=macd ID=outlier_6_1 >> ./logs/outlier_6_1.log
echo 'Finished execution'

echo 'Running tests for Outlier #7'
../bin/python run_ppo.py NUMIT=40000000 LRATE=0.003 LAYERS=v3 ENV=macd ID=outlier_7_1 >> ./logs/outlier_7_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=17500000 LRATE=0.003 LAYERS=v3 ENV=macd ID=outlier_7_2 >> ./logs/outlier_7_2.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=5000000 LRATE=0.0072 LAYERS=v3 ENV=macd ID=outlier_7_3 >> ./logs/outlier_7_3.log
echo 'Finished execution'

echo 'Running tests for Outlier #8'
../bin/python run_ppo.py NUMIT=40000000 LRATE=0.0044 LAYERS=v2 ENV=rsi ID=outlier_8_1 >> ./logs/outlier_8_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=50000000 LRATE=0.0044 LAYERS=v2 ENV=rsi ID=outlier_8_2 >> ./logs/outlier_8_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #9'
../bin/python run_ppo.py NUMIT=10000000 LRATE=0.003 LAYERS=v4 ENV=rsi ID=outlier_9_1 >> ./logs/outlier_9_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0024 LAYERS=v4 ENV=rsi ID=outlier_9_2 >> ./logs/outlier_9_2.log
echo 'Finished execution'