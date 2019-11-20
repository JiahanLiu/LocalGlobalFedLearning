for((c=0; c <2; c++))
do
    python3 search_hyperparameter.py -g 0 &
    python3 search_hyperparameter.py -g 1 & 
    python3 search_hyperparameter.py -g 2 & 
done

