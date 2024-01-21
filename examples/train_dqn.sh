# !bin/sh

echo 'Will start to run through each execution'

echo 'Running tests for Test #1'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.003 LAYERS=v1 ENV=macd ID=test_1_1 >> ./logs/test_1_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0024 LAYERS=v1 ENV=macd ID=test_1_2 >> ./logs/test_1_2.log
echo 'Finished execution'

echo 'Running tests for Test #2'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0092 LAYERS=v2 ENV=macd ID=test_2_1 >> ./logs/test_2_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.01 LAYERS=v2 ENV=macd ID=test_2_2 >> ./logs/test_2_2.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0086 LAYERS=v5 ENV=macd ID=test_2_3 >> ./logs/test_2_3.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0086 LAYERS=v6 ENV=macd ID=test_2_4 >> ./logs/test_2_4.log
echo 'Finished execution'

echo 'Running tests for Test #3'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0044 LAYERS=v3 ENV=macd ID=test_3_1 >> ./logs/test_3_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0041 LAYERS=v3 ENV=macd ID=test_3_2 >> ./logs/test_3_2.log
echo 'Finished execution'

echo 'Running tests for Test #4'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0051 LAYERS=v4 ENV=macd ID=test_4_1 >> ./logs/test_4_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0048 LAYERS=v4 ENV=macd ID=test_4_2 >> ./logs/test_4_2.log
echo 'Finished execution'

echo 'Running tests for Test #5'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0023 LAYERS=v3 ENV=mas ID=test_5_1 >> ./logs/test_5_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.002 LAYERS=v3 ENV=mas ID=test_5_2 >> ./logs/test_5_2.log
echo 'Finished execution'

echo 'Running tests for Test #6'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.007 LAYERS=v2 ENV=rsi ID=test_6_1 >> ./logs/test_6_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0073 LAYERS=v2 ENV=rsi ID=test_6_2 >> ./logs/test_6_2.log
echo 'Finished execution'

echo 'Running tests for Test #7'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0024 LAYERS=v4 ENV=rsi ID=test_7_1 >> ./logs/test_7_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0021 LAYERS=v4 ENV=rsi ID=test_7_2 >> ./logs/test_7_2.log
echo 'Finished execution'

echo 'Running tests for Test #8'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.015 LAYERS=v2 ENV=macd ID=test_8_1 >> ./logs/test_8_1.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.0097 LAYERS=v2 ENV=macd ID=test_8_2 >> ./logs/test_8_2.log
echo 'Finished execution'
../bin/python run_dqn.py NUMIT=500000 LRATE=0.011 LAYERS=v2 ENV=macd ID=test_8_3 >> ./logs/test_8_3.log
echo 'Finished execution'
