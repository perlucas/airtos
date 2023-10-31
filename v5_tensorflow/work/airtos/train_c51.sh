# !bin/sh

echo 'Will start to run through each execution'

echo 'Running tests for Test #1'
../bin/python run_c51.py NUMIT=10000 LRATE=0.0058 LAYERS=v2 ENV=adx ID=test_1_1 >> ./logs/test_1_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=15000 LRATE=0.0058 LAYERS=v2 ENV=adx ID=test_1_2 >> ./logs/test_1_2.log
echo 'Finished execution'

echo 'Running tests for Test #2'
../bin/python run_c51.py NUMIT=15000 LRATE=0.0086 LAYERS=v4 ENV=adx ID=test_2_1 >> ./logs/test_2_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=15000 LRATE=0.006 LAYERS=v4 ENV=adx ID=test_2_2 >> ./logs/test_2_2.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=15000 LRATE=0.005 LAYERS=v4 ENV=adx ID=test_2_3 >> ./logs/test_2_3.log
echo 'Finished execution'

echo 'Running tests for Test #3'
../bin/python run_c51.py NUMIT=1000 LRATE=0.0093 LAYERS=v3 ENV=macd ID=test_3_1 >> ./logs/test_3_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=500 LRATE=0.0093 LAYERS=v3 ENV=macd ID=test_3_2 >> ./logs/test_3_2.log
echo 'Finished execution'

echo 'Running tests for Test #4'
../bin/python run_c51.py NUMIT=40000 LRATE=0.006 LAYERS=v4 ENV=macd ID=test_4_1 >> ./logs/test_4_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=45000 LRATE=0.006 LAYERS=v4 ENV=macd ID=test_4_2 >> ./logs/test_4_2.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=30000 LRATE=0.0064 LAYERS=v4 ENV=macd ID=test_4_3 >> ./logs/test_4_3.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=35000 LRATE=0.0064 LAYERS=v4 ENV=macd ID=test_4_4 >> ./logs/test_4_4.log
echo 'Finished execution'

echo 'Running tests for Test #5'
../bin/python run_c51.py NUMIT=10000 LRATE=0.003 LAYERS=v1 ENV=adx ID=test_5_1 >> ./logs/test_5_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=15000 LRATE=0.003 LAYERS=v1 ENV=adx ID=test_5_2 >> ./logs/test_5_2.log
echo 'Finished execution'

echo 'Running Additional Tests'
../bin/python run_c51.py NUMIT=6000 LRATE=0.006 LAYERS=v2 ENV=adx ID=extra_1 >> ./logs/extra_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=6000 LRATE=0.0096 LAYERS=v3 ENV=macd ID=extra_2 >> ./logs/extra_2.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=6000 LRATE=0.007 LAYERS=v4 ENV=macd ID=extra_3 >> ./logs/extra_3.log
echo 'Finished execution'