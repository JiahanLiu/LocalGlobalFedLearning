for((c=0; c <8; c++))
do
    time (python3 search_hyperparameter.py -g 0 > /dev/null) & 
    time (python3 search_hyperparameter.py -g 1 > /dev/null) & 
    time (python3 search_hyperparameter.py -g 2 > /dev/null) & 
    time (python3 search_hyperparameter.py -g 3 > /dev/null) & 
done

