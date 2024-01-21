# !bin/sh

echo 'Will start to run through each execution'

echo 'Running tests for Test #1'
../bin/python run_ppo.py NUMIT=4000000 LRATE=0.004 LAYERS=v1 ENV=adx ID=test_1_1 >> ./logs/test_1_1.log
echo 'Finished execution' 
../bin/python run_ppo.py NUMIT=4000000 LRATE=0.0036 LAYERS=v1 ENV=adx ID=test_1_2 >> ./logs/test_1_2.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=4000000 LRATE=0.0033 LAYERS=v1 ENV=adx ID=test_1_3 >> ./logs/test_1_3.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=4000000 LRATE=0.0029 LAYERS=v1 ENV=adx ID=test_1_4 >> ./logs/test_1_4.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=4000000 LRATE=0.0025 LAYERS=v1 ENV=adx ID=test_1_5 >> ./logs/test_1_5.log
echo 'Finished execution'

echo 'Running tests for Test #2'
../bin/python run_ppo.py NUMIT=5000000 LRATE=0.005 LAYERS=v2 ENV=adx ID=test_2_1 >> ./logs/test_2_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=5000000 LRATE=0.0045 LAYERS=v2 ENV=adx ID=test_2_2 >> ./logs/test_2_2.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=5000000 LRATE=0.004 LAYERS=v2 ENV=adx ID=test_2_3 >> ./logs/test_2_3.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=5000000 LRATE=0.0035 LAYERS=v2 ENV=adx ID=test_2_4 >> ./logs/test_2_4.log
echo 'Finished execution'

echo 'Running tests for Test #3'
../bin/python run_ppo.py NUMIT=4000000 LRATE=0.0027 LAYERS=v3 ENV=adx ID=test_3_1 >> ./logs/test_3_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=4000000 LRATE=0.0024 LAYERS=v3 ENV=adx ID=test_3_2 >> ./logs/test_3_2.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=4000000 LRATE=0.002 LAYERS=v3 ENV=adx ID=test_3_3 >> ./logs/test_3_3.log
echo 'Finished execution'

echo 'Running tests for Test #4'
../bin/python run_ppo.py NUMIT=3500000 LRATE=0.0027 LAYERS=v4 ENV=adx ID=test_4_1 >> ./logs/test_4_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=3500000 LRATE=0.0024 LAYERS=v4 ENV=adx ID=test_4_2 >> ./logs/test_4_2.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=3500000 LRATE=0.002 LAYERS=v4 ENV=adx ID=test_4_3 >> ./logs/test_4_3.log
echo 'Finished execution'

echo 'Running tests for Test #5'
../bin/python run_ppo.py NUMIT=3000000 LRATE=0.007 LAYERS=v1 ENV=macd ID=test_5_1 >> ./logs/test_5_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=3000000 LRATE=0.0066 LAYERS=v1 ENV=macd ID=test_5_2 >> ./logs/test_5_2.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=3000000 LRATE=0.0063 LAYERS=v1 ENV=macd ID=test_5_3 >> ./logs/test_5_3.log
echo 'Finished execution'

echo 'Running tests for Test #6'
../bin/python run_ppo.py NUMIT=15000000 LRATE=0.0028 LAYERS=v2 ENV=macd ID=test_6_1 >> ./logs/test_6_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=15000000 LRATE=0.0026 LAYERS=v2 ENV=macd ID=test_6_2 >> ./logs/test_6_2.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=15000000 LRATE=0.0024 LAYERS=v2 ENV=macd ID=test_6_3 >> ./logs/test_6_3.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=15000000 LRATE=0.0022 LAYERS=v2 ENV=macd ID=test_6_4 >> ./logs/test_6_4.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=15000000 LRATE=0.002 LAYERS=v2 ENV=macd ID=test_6_5 >> ./logs/test_6_5.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=15000000 LRATE=0.0018 LAYERS=v2 ENV=macd ID=test_6_6 >> ./logs/test_6_6.log
echo 'Finished execution'

echo 'Running tests for Test #7'
../bin/python run_ppo.py NUMIT=9000000 LRATE=0.0027 LAYERS=v3 ENV=macd ID=test_7_1 >> ./logs/test_7_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=9000000 LRATE=0.0025 LAYERS=v3 ENV=macd ID=test_7_2 >> ./logs/test_7_2.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=9000000 LRATE=0.0023 LAYERS=v3 ENV=macd ID=test_7_3 >> ./logs/test_7_3.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=9000000 LRATE=0.002 LAYERS=v3 ENV=macd ID=test_7_4 >> ./logs/test_7_4.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=9000000 LRATE=0.0017 LAYERS=v3 ENV=macd ID=test_7_5 >> ./logs/test_7_5.log
echo 'Finished execution'

echo 'Running tests for Test #8'
../bin/python run_ppo.py NUMIT=10000000 LRATE=0.0037 LAYERS=v2 ENV=rsi ID=test_8_1 >> ./logs/test_8_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=10000000 LRATE=0.0034 LAYERS=v2 ENV=rsi ID=test_8_2 >> ./logs/test_8_2.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=10000000 LRATE=0.003 LAYERS=v2 ENV=rsi ID=test_8_3 >> ./logs/test_8_3.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=10000000 LRATE=0.0026 LAYERS=v2 ENV=rsi ID=test_8_4 >> ./logs/test_8_4.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=10000000 LRATE=0.0023 LAYERS=v2 ENV=rsi ID=test_8_5 >> ./logs/test_8_5.log
echo 'Finished execution'

echo 'Running tests for Test #9'
../bin/python run_ppo.py NUMIT=10000000 LRATE=0.0021 LAYERS=v4 ENV=rsi ID=test_9_1 >> ./logs/test_9_1.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=10000000 LRATE=0.0018 LAYERS=v4 ENV=rsi ID=test_9_2 >> ./logs/test_9_2.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=10000000 LRATE=0.0015 LAYERS=v4 ENV=rsi ID=test_9_3 >> ./logs/test_9_3.log
echo 'Finished execution'
../bin/python run_ppo.py NUMIT=10000000 LRATE=0.0012 LAYERS=v4 ENV=rsi ID=test_9_4 >> ./logs/test_9_4.log
echo 'Finished execution'
