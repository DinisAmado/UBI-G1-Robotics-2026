#!/bin/bash

# Executa cada script no seu caminho próprio, em simultâneo e sem logs
python3 "/caminho/do/script/um/script1.py" > /dev/null 2>&1 &
python3 "/outro/caminho/qualquer/script2.py" > /dev/null 2>&1 &
python3 "/pasta/do/projeto/script3.py" > /dev/null 2>&1 &
python3 "/var/www/scripts/script4.py" > /dev/null 2>&1 &
python3 "/home/user/bot/script5.py" > /dev/null 2>&1 &
python3 "/caminho/do/script/seis/script6.py" > /dev/null 2>&1 &
python3 "/caminho/do/script/sete/script7.py" > /dev/null 2>&1 &

# Espera que todos os 7 scripts terminem
wait
