# !bin/sh

# Use this long time running script to automate execution of each test based on proper combinationn of learning parameters
# Each script will command the REINFORCE agent to run a learning & testing cycle using the given parameters and generate
# a log file with the outcomes of that specific execution

echo 'Will start to run through each execution'

echo 'Running tests for Test #1'
../bin/python run_reinforce.py NUMIT=600 LRATE=0.003 LAYERS=v1 ENV=macd ID=test_1_1 >> ./logs/test_1_1.log
echo 'Finished execution' 
../bin/python run_reinforce.py NUMIT=1200 LRATE=0.0024 LAYERS=v1 ENV=macd ID=test_1_2 >> ./logs/test_1_2.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1200 LRATE=0.002 LAYERS=v1 ENV=macd ID=test_1_3 >> ./logs/test_1_3.log
echo 'Finished execution'

echo 'Running tests for Test #2'
../bin/python run_reinforce.py NUMIT=1500 LRATE=0.017 LAYERS=v3 ENV=mas ID=test_2_1 >> ./logs/test_2_1.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1500 LRATE=0.02 LAYERS=v3 ENV=mas ID=test_2_2 >> ./logs/test_2_2.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1500 LRATE=0.024 LAYERS=v3 ENV=mas ID=test_2_3 >> ./logs/test_2_3.log
echo 'Finished execution'

echo 'Running tests for Test #3'
../bin/python run_reinforce.py NUMIT=1000 LRATE=0.0085 LAYERS=v4 ENV=mas ID=test_3_1 >> ./logs/test_3_1.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1000 LRATE=0.008 LAYERS=v4 ENV=mas ID=test_3_2 >> ./logs/test_3_2.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1000 LRATE=0.0076 LAYERS=v4 ENV=mas ID=test_3_3 >> ./logs/test_3_3.log
echo 'Finished execution'

echo 'Running tests for Test #4'
../bin/python run_reinforce.py NUMIT=1000 LRATE=0.01 LAYERS=v1 ENV=macd ID=test_4_1 >> ./logs/test_4_1.log
echo 'Finished execution'
../bin/python run_reinforce.py NUMIT=1000 LRATE=0.095 LAYERS=v1 ENV=macd ID=test_4_2 >> ./logs/test_4_2.log
echo 'Finished execution'
