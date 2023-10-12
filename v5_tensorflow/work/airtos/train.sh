# !bin/sh

echo 'Will start to run through each execution'

echo 'Total executions found: 80'
echo ''
echo 'Running execution 1 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v1 ENV=macd >> ./logs/run_1.log
echo 'Finished execution'

echo 'Running execution 2 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v2 ENV=macd >> ./logs/run_2.log
echo 'Finished execution'

echo 'Running execution 3 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v3 ENV=macd >> ./logs/run_3.log
echo 'Finished execution'

echo 'Running execution 4 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v4 ENV=macd >> ./logs/run_4.log
echo 'Finished execution'

echo 'Running execution 5 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v1 ENV=rsi >> ./logs/run_5.log
echo 'Finished execution'

echo 'Running execution 6 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v2 ENV=rsi >> ./logs/run_6.log
echo 'Finished execution'

echo 'Running execution 7 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v3 ENV=rsi >> ./logs/run_7.log
echo 'Finished execution'

echo 'Running execution 8 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v4 ENV=rsi >> ./logs/run_8.log
echo 'Finished execution'

echo 'Running execution 9 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v1 ENV=adx >> ./logs/run_9.log
echo 'Finished execution'

echo 'Running execution 10 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v2 ENV=adx >> ./logs/run_10.log
echo 'Finished execution'

echo 'Running execution 11 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v3 ENV=adx >> ./logs/run_11.log
echo 'Finished execution'

echo 'Running execution 12 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v4 ENV=adx >> ./logs/run_12.log
echo 'Finished execution'

echo 'Running execution 13 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v1 ENV=mas >> ./logs/run_13.log
echo 'Finished execution'

echo 'Running execution 14 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v2 ENV=mas >> ./logs/run_14.log
echo 'Finished execution'

echo 'Running execution 15 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v3 ENV=mas >> ./logs/run_15.log
echo 'Finished execution'

echo 'Running execution 16 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.003 LAYERS=v4 ENV=mas >> ./logs/run_16.log
echo 'Finished execution'

echo 'Running execution 17 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v1 ENV=macd >> ./logs/run_17.log
echo 'Finished execution'

echo 'Running execution 18 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v2 ENV=macd >> ./logs/run_18.log
echo 'Finished execution'

echo 'Running execution 19 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v3 ENV=macd >> ./logs/run_19.log
echo 'Finished execution'

echo 'Running execution 20 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v4 ENV=macd >> ./logs/run_20.log
echo 'Finished execution'

echo 'Running execution 21 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v1 ENV=rsi >> ./logs/run_21.log
echo 'Finished execution'

echo 'Running execution 22 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v2 ENV=rsi >> ./logs/run_22.log
echo 'Finished execution'

echo 'Running execution 23 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v3 ENV=rsi >> ./logs/run_23.log
echo 'Finished execution'

echo 'Running execution 24 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v4 ENV=rsi >> ./logs/run_24.log
echo 'Finished execution'

echo 'Running execution 25 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v1 ENV=adx >> ./logs/run_25.log
echo 'Finished execution'

echo 'Running execution 26 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v2 ENV=adx >> ./logs/run_26.log
echo 'Finished execution'

echo 'Running execution 27 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v3 ENV=adx >> ./logs/run_27.log
echo 'Finished execution'

echo 'Running execution 28 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v4 ENV=adx >> ./logs/run_28.log
echo 'Finished execution'

echo 'Running execution 29 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v1 ENV=mas >> ./logs/run_29.log
echo 'Finished execution'

echo 'Running execution 30 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v2 ENV=mas >> ./logs/run_30.log
echo 'Finished execution'

echo 'Running execution 31 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v3 ENV=mas >> ./logs/run_31.log
echo 'Finished execution'

echo 'Running execution 32 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0044 LAYERS=v4 ENV=mas >> ./logs/run_32.log
echo 'Finished execution'

echo 'Running execution 33 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v1 ENV=macd >> ./logs/run_33.log
echo 'Finished execution'

echo 'Running execution 34 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v2 ENV=macd >> ./logs/run_34.log
echo 'Finished execution'

echo 'Running execution 35 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v3 ENV=macd >> ./logs/run_35.log
echo 'Finished execution'

echo 'Running execution 36 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v4 ENV=macd >> ./logs/run_36.log
echo 'Finished execution'

echo 'Running execution 37 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v1 ENV=rsi >> ./logs/run_37.log
echo 'Finished execution'

echo 'Running execution 38 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v2 ENV=rsi >> ./logs/run_38.log
echo 'Finished execution'

echo 'Running execution 39 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v3 ENV=rsi >> ./logs/run_39.log
echo 'Finished execution'

echo 'Running execution 40 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v4 ENV=rsi >> ./logs/run_40.log
echo 'Finished execution'

echo 'Running execution 41 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v1 ENV=adx >> ./logs/run_41.log
echo 'Finished execution'

echo 'Running execution 42 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v2 ENV=adx >> ./logs/run_42.log
echo 'Finished execution'

echo 'Running execution 43 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v3 ENV=adx >> ./logs/run_43.log
echo 'Finished execution'

echo 'Running execution 44 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v4 ENV=adx >> ./logs/run_44.log
echo 'Finished execution'

echo 'Running execution 45 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v1 ENV=mas >> ./logs/run_45.log
echo 'Finished execution'

echo 'Running execution 46 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v2 ENV=mas >> ./logs/run_46.log
echo 'Finished execution'

echo 'Running execution 47 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v3 ENV=mas >> ./logs/run_47.log
echo 'Finished execution'

echo 'Running execution 48 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0058 LAYERS=v4 ENV=mas >> ./logs/run_48.log
echo 'Finished execution'

echo 'Running execution 49 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v1 ENV=macd >> ./logs/run_49.log
echo 'Finished execution'

echo 'Running execution 50 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v2 ENV=macd >> ./logs/run_50.log
echo 'Finished execution'

echo 'Running execution 51 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v3 ENV=macd >> ./logs/run_51.log
echo 'Finished execution'

echo 'Running execution 52 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v4 ENV=macd >> ./logs/run_52.log
echo 'Finished execution'

echo 'Running execution 53 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v1 ENV=rsi >> ./logs/run_53.log
echo 'Finished execution'

echo 'Running execution 54 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v2 ENV=rsi >> ./logs/run_54.log
echo 'Finished execution'

echo 'Running execution 55 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v3 ENV=rsi >> ./logs/run_55.log
echo 'Finished execution'

echo 'Running execution 56 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v4 ENV=rsi >> ./logs/run_56.log
echo 'Finished execution'

echo 'Running execution 57 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v1 ENV=adx >> ./logs/run_57.log
echo 'Finished execution'

echo 'Running execution 58 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v2 ENV=adx >> ./logs/run_58.log
echo 'Finished execution'

echo 'Running execution 59 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v3 ENV=adx >> ./logs/run_59.log
echo 'Finished execution'

echo 'Running execution 60 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v4 ENV=adx >> ./logs/run_60.log
echo 'Finished execution'

echo 'Running execution 61 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v1 ENV=mas >> ./logs/run_61.log
echo 'Finished execution'

echo 'Running execution 62 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v2 ENV=mas >> ./logs/run_62.log
echo 'Finished execution'

echo 'Running execution 63 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v3 ENV=mas >> ./logs/run_63.log
echo 'Finished execution'

echo 'Running execution 64 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0072 LAYERS=v4 ENV=mas >> ./logs/run_64.log
echo 'Finished execution'

echo 'Running execution 65 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v1 ENV=macd >> ./logs/run_65.log
echo 'Finished execution'

echo 'Running execution 66 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v2 ENV=macd >> ./logs/run_66.log
echo 'Finished execution'

echo 'Running execution 67 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v3 ENV=macd >> ./logs/run_67.log
echo 'Finished execution'

echo 'Running execution 68 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v4 ENV=macd >> ./logs/run_68.log
echo 'Finished execution'

echo 'Running execution 69 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v1 ENV=rsi >> ./logs/run_69.log
echo 'Finished execution'

echo 'Running execution 70 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v2 ENV=rsi >> ./logs/run_70.log
echo 'Finished execution'

echo 'Running execution 71 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v3 ENV=rsi >> ./logs/run_71.log
echo 'Finished execution'

echo 'Running execution 72 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v4 ENV=rsi >> ./logs/run_72.log
echo 'Finished execution'

echo 'Running execution 73 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v1 ENV=adx >> ./logs/run_73.log
echo 'Finished execution'

echo 'Running execution 74 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v2 ENV=adx >> ./logs/run_74.log
echo 'Finished execution'

echo 'Running execution 75 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v3 ENV=adx >> ./logs/run_75.log
echo 'Finished execution'

echo 'Running execution 76 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v4 ENV=adx >> ./logs/run_76.log
echo 'Finished execution'

echo 'Running execution 77 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v1 ENV=mas >> ./logs/run_77.log
echo 'Finished execution'

echo 'Running execution 78 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v2 ENV=mas >> ./logs/run_78.log
echo 'Finished execution'

echo 'Running execution 79 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v3 ENV=mas >> ./logs/run_79.log
echo 'Finished execution'

echo 'Running execution 80 of 80'
../bin/python run_ppo.py NUMIT=25000000 LRATE=0.0086 LAYERS=v4 ENV=mas >> ./logs/run_80.log
echo 'Finished execution'
