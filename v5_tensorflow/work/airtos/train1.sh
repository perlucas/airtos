# !bin/sh

echo 'Will start to run through each execution'

echo 'Total executions found: 80'
echo ''
echo 'Running execution 1 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v1 ENV=macd >> ./logs/run_1.log
echo 'Finished execution'

echo 'Running execution 2 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v2 ENV=macd >> ./logs/run_2.log
echo 'Finished execution'

echo 'Running execution 3 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v3 ENV=macd >> ./logs/run_3.log
echo 'Finished execution'

echo 'Running execution 4 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v4 ENV=macd >> ./logs/run_4.log
echo 'Finished execution'

echo 'Running execution 5 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v1 ENV=rsi >> ./logs/run_5.log
echo 'Finished execution'

echo 'Running execution 6 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v2 ENV=rsi >> ./logs/run_6.log
echo 'Finished execution'

echo 'Running execution 7 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v3 ENV=rsi >> ./logs/run_7.log
echo 'Finished execution'

echo 'Running execution 8 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v4 ENV=rsi >> ./logs/run_8.log
echo 'Finished execution'

echo 'Running execution 9 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v1 ENV=adx >> ./logs/run_9.log
echo 'Finished execution'

echo 'Running execution 10 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v2 ENV=adx >> ./logs/run_10.log
echo 'Finished execution'

echo 'Running execution 11 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v3 ENV=adx >> ./logs/run_11.log
echo 'Finished execution'

echo 'Running execution 12 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v4 ENV=adx >> ./logs/run_12.log
echo 'Finished execution'

echo 'Running execution 13 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v1 ENV=mas >> ./logs/run_13.log
echo 'Finished execution'

echo 'Running execution 14 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v2 ENV=mas >> ./logs/run_14.log
echo 'Finished execution'

echo 'Running execution 15 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v3 ENV=mas >> ./logs/run_15.log
echo 'Finished execution'

echo 'Running execution 16 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.003 LAYERS=v4 ENV=mas >> ./logs/run_16.log
echo 'Finished execution'

echo 'Running execution 17 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.0044 LAYERS=v1 ENV=macd >> ./logs/run_17.log
echo 'Finished execution'

echo 'Running execution 18 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.0044 LAYERS=v2 ENV=macd >> ./logs/run_18.log
echo 'Finished execution'

echo 'Running execution 19 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.0044 LAYERS=v3 ENV=macd >> ./logs/run_19.log
echo 'Finished execution'

echo 'Running execution 20 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.0044 LAYERS=v4 ENV=macd >> ./logs/run_20.log
echo 'Finished execution'

echo 'Running execution 21 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.0044 LAYERS=v1 ENV=rsi >> ./logs/run_21.log
echo 'Finished execution'

echo 'Running execution 22 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.0044 LAYERS=v2 ENV=rsi >> ./logs/run_22.log
echo 'Finished execution'

echo 'Running execution 23 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.0044 LAYERS=v3 ENV=rsi >> ./logs/run_23.log
echo 'Finished execution'

echo 'Running execution 24 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.0044 LAYERS=v4 ENV=rsi >> ./logs/run_24.log
echo 'Finished execution'

echo 'Running execution 25 of 80'
../bin/python run_c51.py NUMIT=200000 LRATE=0.0044 LAYERS=v1 ENV=adx >> ./logs/run_25.log
echo 'Finished execution'
