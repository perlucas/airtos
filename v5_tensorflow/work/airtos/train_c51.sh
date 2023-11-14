# !bin/sh

echo 'Will start to run through each execution'

echo 'Running tests for Test #1'
../bin/python run_c51.py NUMIT=30000 ENV=macd LAYERS=v4 LRATE=0.0058 ID=test_1_1 >> ./logs/test_1_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=30000 ENV=macd LAYERS=v4 LRATE=0.0057 ID=test_1_2 >> ./logs/test_1_2.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=30000 ENV=macd LAYERS=v4 LRATE=0.0069 ID=test_1_3 >> ./logs/test_1_3.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=30000 ENV=macd LAYERS=v4 LRATE=0.007 ID=test_1_4 >> ./logs/test_1_4.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=30000 ENV=macd LAYERS=v5 LRATE=0.007 ID=test_1_5 >> ./logs/test_1_5.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=30000 ENV=macd LAYERS=v6 LRATE=0.007 ID=test_1_6 >> ./logs/test_1_6.log
echo 'Finished execution'

echo 'Running tests for Test #2'
../bin/python run_c51.py NUMIT=4000 ENV=macd LAYERS=v3 LRATE=0.0095 ID=test_2_1 >> ./logs/test_2_1.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=4000 ENV=macd LAYERS=v3 LRATE=0.0097 ID=test_2_2 >> ./logs/test_2_2.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=3000 ENV=macd LAYERS=v4 LRATE=0.0075 ID=test_2_3 >> ./logs/test_2_3.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=3000 ENV=macd LAYERS=v4 LRATE=0.0078 ID=test_2_4 >> ./logs/test_2_4.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=3000 ENV=macd LAYERS=v5 LRATE=0.0075 ID=test_2_5 >> ./logs/test_2_5.log
echo 'Finished execution'
../bin/python run_c51.py NUMIT=3000 ENV=macd LAYERS=v6 LRATE=0.0075 ID=test_2_6 >> ./logs/test_2_6.log
echo 'Finished execution'
