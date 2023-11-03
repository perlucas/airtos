# !bin/sh

echo 'Will start to run through each execution'

echo 'Running tests for Test #1'
../bin/python run_c51.py NUMIT=15000 LRATE=0.006 LAYERS=v4 ENV=adx ID=test_1_1 >> ./logs/test_1_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=15000 LRATE=0.0065 LAYERS=v4 ENV=adx ID=test_1_2 >> ./logs/test_1_2.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=15000 LRATE=0.0058 LAYERS=v4 ENV=adx ID=test_1_3 >> ./logs/test_1_3.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=15000 LRATE=0.007 LAYERS=v4 ENV=adx ID=test_1_4 >> ./logs/test_1_4.log
echo 'Finished execution'

echo 'Running tests for Test #2'
../bin/python run_c51.py NUMIT=1000 LRATE=0.0093 LAYERS=v3 ENV=macd ID=test_2_1 >> ./logs/test_2_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=1000 LRATE=0.009 LAYERS=v3 ENV=macd ID=test_2_2 >> ./logs/test_2_2.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=1000 LRATE=0.012 LAYERS=v3 ENV=macd ID=test_2_3 >> ./logs/test_2_3.log
echo 'Finished execution'

echo 'Running tests for Test #3'
../bin/python run_c51.py NUMIT=30000 LRATE=0.006 LAYERS=v4 ENV=macd ID=test_3_1 >> ./logs/test_3_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=30000 LRATE=0.0064 LAYERS=v4 ENV=macd ID=test_3_2 >> ./logs/test_3_2.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=30000 LRATE=0.0062 LAYERS=v4 ENV=macd ID=test_3_3 >> ./logs/test_3_3.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=30000 LRATE=0.0067 LAYERS=v4 ENV=macd ID=test_3_4 >> ./logs/test_3_4.log
echo 'Finished execution'

echo 'Running tests for Test #4'
../bin/python run_c51.py NUMIT=4000 LRATE=0.006 LAYERS=v2 ENV=adx ID=test_4_1 >> ./logs/test_4_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=4000 LRATE=0.0057 LAYERS=v2 ENV=adx ID=test_4_2 >> ./logs/test_4_2.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=4000 LRATE=0.0064 LAYERS=v2 ENV=adx ID=test_4_3 >> ./logs/test_4_3.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=4000 LRATE=0.0096 LAYERS=v3 ENV=macd ID=test_4_4 >> ./logs/test_4_4.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=4000 LRATE=0.011 LAYERS=v3 ENV=macd ID=test_4_5 >> ./logs/test_4_5.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=4000 LRATE=0.0092 LAYERS=v3 ENV=macd ID=test_4_6 >> ./logs/test_4_6.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=3000 LRATE=0.007 LAYERS=v4 ENV=macd ID=test_4_7 >> ./logs/test_4_7.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=3000 LRATE=0.0074 LAYERS=v4 ENV=macd ID=test_4_8 >> ./logs/test_4_8.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=3000 LRATE=0.0067 LAYERS=v4 ENV=macd ID=test_4_9 >> ./logs/test_4_9.log
echo 'Finished execution'
