# !bin/sh

echo 'Will start to run through each execution'

echo 'Running tests for Outlier #1'
../bin/python run_dqn.py NUMIT=600000 LRATE=0.003 LAYERS=v1 ENV=macd ID=outlier_1_1 >> ./logs/outlier_1_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=600000 LRATE=0.002 LAYERS=v1 ENV=macd ID=outlier_1_2 >> ./logs/outlier_1_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #2'
../bin/python run_dqn.py NUMIT=370000 LRATE=0.0086 LAYERS=v2 ENV=macd ID=outlier_2_1 >> ./logs/outlier_2_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0092 LAYERS=v2 ENV=macd ID=outlier_2_2 >> ./logs/outlier_2_2.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.01 LAYERS=v2 ENV=macd ID=outlier_2_3 >> ./logs/outlier_2_3.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0086 LAYERS=v5 ENV=macd ID=outlier_2_4 >> ./logs/outlier_2_4.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0086 LAYERS=v6 ENV=macd ID=outlier_2_5 >> ./logs/outlier_2_5.log
echo 'Finished execution'

echo 'Running tests for Outlier #3'
../bin/python run_dqn.py NUMIT=200000 LRATE=0.0044 LAYERS=v3 ENV=macd ID=outlier_3_1 >> ./logs/outlier_3_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0037 LAYERS=v3 ENV=macd ID=outlier_3_2 >> ./logs/outlier_3_2.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0051 LAYERS=v3 ENV=macd ID=outlier_3_3 >> ./logs/outlier_3_3.log
echo 'Finished execution'

echo 'Running tests for Outlier #4'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0051 LAYERS=v4 ENV=macd ID=outlier_4_1 >> ./logs/outlier_4_1.log
echo 'Finished execution'

echo 'Running tests for Outlier #5'
../bin/python run_dqn.py NUMIT=600000 LRATE=0.002 LAYERS=v3 ENV=mas ID=outlier_5_1 >> ./logs/outlier_5_1.log
echo 'Finished execution'

echo 'Running tests for Outlier #6'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.005 LAYERS=v1 ENV=rsi ID=outlier_6_1 >> ./logs/outlier_6_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0054 LAYERS=v1 ENV=rsi ID=outlier_6_2 >> ./logs/outlier_6_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #7'
../bin/python run_dqn.py NUMIT=800000 LRATE=0.007 LAYERS=v2 ENV=rsi ID=outlier_7_1 >> ./logs/outlier_7_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=600000 LRATE=0.0065 LAYERS=v2 ENV=rsi ID=outlier_7_2 >> ./logs/outlier_7_2.log
echo 'Finished execution'

echo 'Running tests for Outlier #8'
../bin/python run_dqn.py NUMIT=600000 LRATE=0.0025 LAYERS=v4 ENV=rsi ID=outlier_8_1 >> ./logs/outlier_8_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=600000 LRATE=0.002 LAYERS=v4 ENV=rsi ID=outlier_8_2 >> ./logs/outlier_8_2.log
echo 'Finished execution'

echo 'Running Additional Tests'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0095 LAYERS=v2 ENV=macd ID=extra_1 >> ./logs/extra_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.015 LAYERS=v2 ENV=macd ID=extra_2 >> ./logs/extra_2.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.03 LAYERS=v2 ENV=macd ID=extra_3 >> ./logs/extra_3.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.07 LAYERS=v2 ENV=macd ID=extra_4 >> ./logs/extra_4.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.1 LAYERS=v2 ENV=macd ID=extra_5 >> ./logs/extra_5.log
echo 'Finished execution'